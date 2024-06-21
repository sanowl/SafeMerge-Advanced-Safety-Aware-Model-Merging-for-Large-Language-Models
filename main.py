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
from typing import List, Tuple, Dict
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer

class SafetyAwareMerger:
    def __init__(self, base_model_name: str, expert_model_names: List[str], device: torch.device, hf_token: str):
        logger.debug(f"Initializing SafetyAwareMerger with base model: {base_model_name}")
        self.device = device
        self.hf_token = hf_token
        self.base_model = self.load_model(base_model_name)
        self.expert_models = [self.load_model(name) for name in expert_model_names]
        self.merged_model = None
        logger.debug("Loading safety classifier")
        self.safety_classifier = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target", device=device)
        logger.debug("SafetyAwareMerger initialized")

    def load_model(self, model_name: str) -> ModelConfig:
        logger.debug(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            use_auth_token=self.hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=self.hf_token
        )
        logger.debug(f"Model {model_name} loaded successfully")
        return ModelConfig(model_name, model, tokenizer)

    def generate_safety_data(self, num_samples: int = 100) -> List[Tuple[str, str]]:
        logger.debug(f"Generating safety data with {num_samples} samples")
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        unsafe_queries = random.sample(dataset["text"], num_samples)
        
        safety_data = []
        for query in tqdm(unsafe_queries, desc="Generating safety data"):
            responses = [self.generate_response(model.model, model.tokenizer, query) 
                         for model in [self.base_model] + self.expert_models]
            safe_responses = [resp for resp in responses if self.is_safe_response(resp)]
            if safe_responses:
                safety_data.append((query, random.choice(safe_responses)))
        
        logger.debug(f"Generated {len(safety_data)} safety data points")
        return safety_data

    def generate_domain_data(self, num_samples: int = 100) -> List[Tuple[str, str, str, str]]:
        logger.debug(f"Generating domain data with {num_samples} samples")
        datasets = [
            load_dataset("bigbio/med_qa", split="train"),
            load_dataset("metaeval/mathqa", split="train"),
        ]
        
        domain_data = []
        for dataset in datasets:
            for item in tqdm(random.sample(dataset, num_samples // len(datasets)), desc=f"Generating data for {dataset.name}"):
                question = item['question']
                answer = item.get('answer', item.get('correct'))
                for model in self.expert_models:
                    response = self.generate_response(model.model, model.tokenizer, question)
                    domain_data.append((question, response, answer, dataset.name))
        
        logger.debug(f"Generated {len(domain_data)} domain data points")
        return domain_data

    @torch.no_grad()
    def generate_response(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, query: str) -> str:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def is_safe_response(self, response: str) -> bool:
        result = self.safety_classifier(response)[0]
        return result['label'] == 'LABEL_0' and result['score'] > 0.9

    def dare_ties_merge(self, lambdas: List[float], sparsity: float = 0.5):
        logger.debug(f"Performing DARE-TIES merge with lambdas: {lambdas}, sparsity: {sparsity}")
        merged_state_dict = {}
        for param_name, base_param in self.base_model.model.named_parameters():
            merged_param = base_param * lambdas[0]
            for i, expert_model in enumerate(self.expert_models):
                expert_param = dict(expert_model.model.named_parameters())[param_name]
                merged_param += expert_param * lambdas[i+1]
            
            mask = torch.rand_like(base_param) < sparsity
            merged_param = torch.where(mask, merged_param, base_param)
            merged_state_dict[param_name] = merged_param
        
        self.merged_model = AutoModelForCausalLM.from_pretrained(
            self.base_model.model.config._name_or_path,
            state_dict=merged_state_dict,
            torch_dtype=torch.float16
        ).to(self.device)
        logger.debug("DARE-TIES merge completed")

    def evaluate_model(self, model: AutoModelForCausalLM, safety_data: List[Tuple[str, str]], domain_data: List[Tuple[str, str, str, str]]) -> Tuple[float, Dict[str, float]]:
        logger.debug("Starting model evaluation")
        safety_score = sum(self.is_safe_response(self.generate_response(model, self.base_model.tokenizer, query))
                           for query, _ in tqdm(safety_data, desc="Evaluating safety"))
        
        domain_scores = {dataset: 0 for dataset in set(data[3] for data in domain_data)}
        for question, _, answer, dataset in tqdm(domain_data, desc="Evaluating domain expertise"):
            response = self.generate_response(model, self.base_model.tokenizer, question)
            if answer.lower() in response.lower():
                domain_scores[dataset] += 1
        
        safety_score /= len(safety_data)
        domain_scores = {k: v / len([d for d in domain_data if d[3] == k]) for k, v in domain_scores.items()}
        logger.debug(f"Evaluation complete. Safety score: {safety_score}, Domain scores: {domain_scores}")
        return safety_score, domain_scores

    def objective_function(self, params: List[float], safety_data: List[Tuple[str, str]], domain_data: List[Tuple[str, str, str, str]]) -> float:
        logger.debug(f"Calculating objective function with params: {params}")
        lambdas, sparsity = params[:-1], params[-1]
        self.dare_ties_merge(lambdas, sparsity)
        safety_score, domain_scores = self.evaluate_model(self.merged_model, safety_data, domain_data)
        objective_value = -(safety_score + sum(domain_scores.values()) / len(domain_scores))
        logger.debug(f"Objective function value: {objective_value}")
        return objective_value

    def evomm_optimize(self, safety_data: List[Tuple[str, str]], domain_data: List[Tuple[str, str, str, str]]) -> List[float]:
        logger.debug("Starting EVOMM optimization")
        bounds = [(0, 1)] * (len(self.expert_models) + 2)  # +1 for base model, +1 for sparsity
        result = differential_evolution(
            partial(self.objective_function, safety_data=safety_data, domain_data=domain_data),
            bounds,
            popsize=20,
            maxiter=100,
            workers=-1  # Use all available CPU cores
        )
        logger.debug(f"EVOMM optimization complete. Optimal params: {result.x}")
        return result.x

def setup(rank: int, world_size: int):
    logger.debug(f"Setting up distributed environment. Rank: {rank}, World Size: {world_size}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    logger.debug("Distributed environment setup complete")

def cleanup():
    logger.debug("Cleaning up distributed environment")
    torch.distributed.destroy_process_group()
    logger.debug("Cleanup complete")

def run_merger(rank: int, world_size: int):
    logger.debug(f"Starting merger run. Rank: {rank}, World Size: {world_size}")
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")
    
    merger = SafetyAwareMerger(
        "gpt2",  # Use a publicly accessible model like GPT-2
        ["microsoft/BioGPT-Large", "EleutherAI/gpt-neox-20b"],
        device,
        hf_token="your_hugging_face_token"  # Include your Hugging Face token here
    )
    
    logger.debug("Generating safety and domain data")
    safety_data = merger.generate_safety_data(num_samples=100)
    domain_data = merger.generate_domain_data(num_samples=100)
    
    # Wrap models in DDP
    logger.debug("Wrapping models in DistributedDataParallel")
    merger.base_model.model = DDP(merger.base_model.model, device_ids=[rank])
    merger.expert_models = [ModelConfig(model.name, DDP(model.model, device_ids=[rank]), model.tokenizer) for model in merger.expert_models]
    
    logger.debug("Starting optimization")
    optimal_params = merger.evomm_optimize(safety_data, domain_data)
    
    logger.debug("Merging models with optimal parameters")
    merger.dare_ties_merge(optimal_params[:-1], optimal_params[-1])
    
    logger.debug("Evaluating merged model")
    safety_score, domain_scores = merger.evaluate_model(merger.merged_model, safety_data, domain_data)
    logger.info(f"Merged Model - Safety: {safety_score:.2f}, Domain Performances: {domain_scores}")
    logger.info(f"Optimal parameters: {optimal_params}")
    
    cleanup()

def run_merger_simple():
    logger.debug("Starting simple merger run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")
    
    merger = SafetyAwareMerger(
        "gpt2",  # Use a publicly accessible model like GPT-2
        ["microsoft/BioGPT-Large", "EleutherAI/gpt-neox-20b"],
        device,
        hf_token=""  # Include your Hugging Face token here
    )
    
    logger.debug("Generating safety and domain data")
    safety_data = merger.generate_safety_data(num_samples=100)
    domain_data = merger.generate_domain_data(num_samples=100)
    
    logger.debug("Starting optimization")
    optimal_params = merger.evomm_optimize(safety_data, domain_data)
    
    logger.debug("Merging models with optimal parameters")
    merger.dare_ties_merge(optimal_params[:-1], optimal_params[-1])
    
    logger.debug("Evaluating merged model")
    safety_score, domain_scores = merger.evaluate_model(merger.merged_model, safety_data, domain_data)
    logger.info(f"Merged Model - Safety: {safety_score:.2f}, Domain Performances: {domain_scores}")
    logger.info(f"Optimal parameters: {optimal_params}")

if __name__ == "__main__":
    logger.debug("Script started")
    world_size = torch.cuda.device_count()
    logger.debug(f"Number of available GPUs: {world_size}")
    if world_size > 1:
        logger.info("Running distributed merger")
        mp.spawn(run_merger, args=(world_size,), nprocs=world_size, join=True)
    else:
        logger.info("Running simple merger")
        run_merger_simple()
    logger.debug("Script completed")
