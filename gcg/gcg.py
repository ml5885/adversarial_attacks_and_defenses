import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Union
from tqdm import tqdm
import copy
import logging

from gcg.nanogcg import GCGConfig, GCGResult, sample_ids_from_grad, filter_ids
from gcg.utils import get_nonascii_toks

logger = logging.getLogger(__name__)

class GCGEnsemble:
    def __init__(self, models: List[PreTrainedModel], tokenizer: PreTrainedTokenizer, config: GCGConfig):
        self.models = models
        self.tokenizer = tokenizer
        self.config = config
        self.device = self.models[0].device
        self.embedding_layers = [model.get_input_embeddings() for model in self.models]
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=self.device)

    def run(self, messages: Union[str, List[dict]], target: str) -> GCGResult:
        config = self.config
        tokenizer = self.tokenizer

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")
        
        target_str = " " + target if config.add_space_before_target else target

        # Prepare embeds for each model
        self.model_contexts = []
        for i, model in enumerate(self.models):
            embedding_layer = self.embedding_layers[i]
            before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(self.device)
            after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
            target_ids = tokenizer([target_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)

            before_embeds = embedding_layer(before_ids)
            after_embeds = embedding_layer(after_ids)
            target_embeds = embedding_layer(target_ids)
            
            prefix_cache = None
            if config.use_prefix_cache:
                with torch.no_grad():
                    output = model(inputs_embeds=before_embeds, use_cache=True)
                    prefix_cache = output.past_key_values
            
            self.model_contexts.append({
                "before_embeds": before_embeds,
                "after_embeds": after_embeds,
                "target_ids": target_ids,
                "target_embeds": target_embeds,
                "prefix_cache": prefix_cache,
                "embedding_layer": embedding_layer
            })

        optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        
        losses = []
        strings = []
        
        for i in tqdm(range(config.num_steps), desc="Ensemble GCG"):
            avg_grad = self.compute_avg_gradient(optim_ids)
            
            with torch.no_grad():
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    avg_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)
                
                candidate_losses = self.compute_avg_candidate_losses(sampled_ids)
                best_candidate_idx = torch.argmin(candidate_losses)
                current_loss = candidate_losses[best_candidate_idx].item()
                optim_ids = sampled_ids[best_candidate_idx].unsqueeze(0)
                
                losses.append(current_loss)
                best_str = tokenizer.decode(optim_ids[0], skip_special_tokens=True)
                strings.append(best_str)

                if (i + 1) % 20 == 0:
                    logger.info(f"Step {i+1}/{config.num_steps}, Loss: {current_loss:.4f}, Suffix: '{best_str}'")

        best_loss_idx = losses.index(min(losses))
        return GCGResult(
            best_loss=losses[best_loss_idx],
            best_string=strings[best_loss_idx],
            losses=losses,
            strings=strings,
        )

    def compute_avg_gradient(self, optim_ids):
        grads = []
        for i, model in enumerate(self.models):
            context = self.model_contexts[i]
            embedding_layer = context["embedding_layer"]

            optim_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings).to(model.dtype)
            optim_onehot.requires_grad_()
            optim_embeds = optim_onehot @ embedding_layer.weight
            
            if self.config.use_prefix_cache:
                input_embeds = torch.cat([optim_embeds, context["after_embeds"], context["target_embeds"]], dim=1)
                outputs = model(inputs_embeds=input_embeds, past_key_values=context["prefix_cache"])
            else:
                input_embeds = torch.cat([context["before_embeds"], optim_embeds, context["after_embeds"], context["target_embeds"]], dim=1)
                outputs = model(inputs_embeds=input_embeds)
            
            logits = outputs.logits
            target_ids = context["target_ids"]
            shift = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., shift - 1 : -1, :].contiguous()
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), target_ids.view(-1))
            
            grad = torch.autograd.grad(loss, [optim_onehot])[0]
            grads.append(grad)
        
        avg_grad = torch.stack(grads).mean(dim=0)
        return avg_grad

    def compute_avg_candidate_losses(self, sampled_ids):
        all_model_losses = []
        num_candidates = sampled_ids.shape[0]

        for i, model in enumerate(self.models):
            context = self.model_contexts[i]
            embedding_layer = context["embedding_layer"]
            candidate_embeds = embedding_layer(sampled_ids)
            
            if self.config.use_prefix_cache:
                input_embeds = torch.cat([candidate_embeds, context["after_embeds"].repeat(num_candidates, 1, 1), context["target_embeds"].repeat(num_candidates, 1, 1)], dim=1)
                
                prefix_cache_batch = [[pc.expand(num_candidates, -1, -1, -1) for pc in layer_cache] for layer_cache in context["prefix_cache"]]
                outputs = model(inputs_embeds=input_embeds, past_key_values=prefix_cache_batch)
            else:
                input_embeds = torch.cat([context["before_embeds"].repeat(num_candidates, 1, 1), candidate_embeds, context["after_embeds"].repeat(num_candidates, 1, 1), context["target_embeds"].repeat(num_candidates, 1, 1)], dim=1)
                outputs = model(inputs_embeds=input_embeds)

            logits = outputs.logits
            target_ids = context["target_ids"]
            shift = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., shift - 1 : -1, :].contiguous()
            
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), target_ids.repeat(num_candidates, 1).view(-1), reduction="none")
            model_losses = loss.view(num_candidates, -1).mean(dim=1)
            all_model_losses.append(model_losses)

        avg_losses = torch.stack(all_model_losses).mean(dim=0)
        return avg_losses

def run(models: List[PreTrainedModel], tokenizer: PreTrainedTokenizer, messages: Union[str, List[dict]], target: str, config: GCGConfig = None) -> GCGResult:
    if config is None:
        config = GCGConfig()
    
    ensemble_gcg = GCGEnsemble(models, tokenizer, config)
    return ensemble_gcg.run(messages, target)