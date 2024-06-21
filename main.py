import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from scipy.optimize import differential_evolution
import random
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyAwareMerger:
    def __init__(self, base_model_name, expert_model_names, device):
        self.device = device
        self.base_model, self.base_tokenizer = self.load_model(base_model_name)
        self.expert_models = [self.load_model(name) for name in expert_model_names]
        self.merged_model = None
        self.safety_classifier = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target", device=device)

    @staticmethod
    def load_model(model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def generate_safety_data(self, num_samples=10000):
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        unsafe_queries = random.sample(dataset["text"], num_samples)
        
        safety_data = []
        for query in tqdm(unsafe_queries, desc="Generating safety data"):
            responses = [self.generate_response(model, tokenizer, query) 
                         for model, tokenizer in [self.base_model, self.base_tokenizer] + self.expert_models]
            safe_responses = [resp for resp in responses if self.is_safe_response(resp)]
            if safe_responses:
                safety_data.append((query, random.choice(safe_responses)))
        
        return safety_data

    def generate_domain_data(self, num_samples=10000):
        datasets = [
            load_dataset("bigbio/med_qa", split="train"),
            load_dataset("metaeval/mathqa", split="train"),
        ]
        
        domain_data = []
        for dataset in datasets:
            for item in tqdm(random.sample(dataset, num_samples // len(datasets)), desc=f"Generating data for {dataset.name}"):
                question = item['question']
                answer = item.get('answer', item.get('correct'))
                for model, tokenizer in self.expert_models:
                    response = self.generate_response(model, tokenizer, question)
                    domain_data.append((question, response, answer, dataset.name))
        
        return domain_data

    @torch.no_grad()
    def generate_response(self, model, tokenizer, query):
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def is_safe_response(self, response):
        result = self.safety_classifier(response)[0]
        return result['label'] == 'LABEL_0' and result['score'] > 0.9

    def dare_ties_merge(self, lambdas, sparsity=0.5):
        merged_state_dict = {}
        for param_name, base_param in self.base_model.named_parameters():
            merged_param = base_param * lambdas[0]
            for i, (expert_model, _) in enumerate(self.expert_models):
                expert_param = dict(expert_model.named_parameters())[param_name]
                merged_param += expert_param * lambdas[i+1]
            
            mask = torch.rand_like(base_param) < sparsity
            merged_param = torch.where(mask, merged_param, base_param)
            merged_state_dict[param_name] = merged_param
        
        self.merged_model = AutoModelForCausalLM.from_pretrained(
            self.base_model.config._name_or_path,
            state_dict=merged_state_dict,
            torch_dtype=torch.float16
        ).to(self.device)

    def evaluate_model(self, model, safety_data, domain_data):
        safety_score = sum(self.is_safe_response(self.generate_response(model, self.base_tokenizer, query))
                           for query, _ in tqdm(safety_data, desc="Evaluating safety"))
        
        domain_scores = {dataset: 0 for dataset in set(data[3] for data in domain_data)}
        for question, _, answer, dataset in tqdm(domain_data, desc="Evaluating domain expertise"):
            response = self.generate_response(model, self.base_tokenizer, question)
            if answer.lower() in response.lower():
                domain_scores[dataset] += 1
        
        safety_score /= len(safety_data)
        domain_scores = {k: v / len([d for d in domain_data if d[3] == k]) for k, v in domain_scores.items()}
        return safety_score, domain_scores

    def objective_function(self, params, safety_data, domain_data):
        lambdas, sparsity = params[:-1], params[-1]
        self.dare_ties_merge(lambdas, sparsity)
        safety_score, domain_scores = self.evaluate_model(self.merged_model, safety_data, domain_data)
        return -(safety_score + sum(domain_scores.values()) / len(domain_scores))

    def evomm_optimize(self, safety_data, domain_data):
        bounds = [(0, 1)] * (len(self.expert_models) + 2)  # +1 for base model, +1 for sparsity
        result = differential_evolution(
            partial(self.objective_function, safety_data=safety_data, domain_data=domain_data),
            bounds,
            popsize=20,
            maxiter=100,
            workers=-1  # Use all available CPU cores
        )
        return result.x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def run_merger(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    merger = SafetyAwareMerger(
        "mistralai/Mistral-7B-Instruct-v0.2",
        ["microsoft/BioGPT-Large", "EleutherAI/gpt-neox-20b"],
        device
    )
    
    safety_data = merger.generate_safety_data(num_samples=1000)
    domain_data = merger.generate_domain_data(num_samples=1000)
    
    # Wrap models in DDP
    merger.base_model = DDP(merger.base_model, device_ids=[rank])
    merger.expert_models = [(DDP(model, device_ids=[rank]), tokenizer) for model, tokenizer in merger.expert_models]
    
    optimal_params = merger.evomm_optimize(safety_data, domain_data)
    
    merger.dare_ties_merge(optimal_params[:-1], optimal_params[-1])
    
    safety_score, domain_scores = merger.evaluate_model(merger.merged_model, safety_data, domain_data)
    logger.info(f"Merged Model - Safety: {safety_score:.2f}, Domain Performances: {domain_scores}")
    logger.info(f"Optimal parameters: {optimal_params}")
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run_merger, args=(world_size,), nprocs=world_size, join=True)