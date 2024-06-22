import argparse
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Dict, Optional
import sagemaker
import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from scipy.optimize import differential_evolution
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, util

# AWS imports
import boto3
import botocore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self, name: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer

class SafetyAwareMerger:
    def __init__(self, base_model_name: str, expert_model_names: List[str], device: torch.device, hf_token: str):
        self.device = device
        self.hf_token = hf_token
        self.base_model = self._load_model(base_model_name)
        self.expert_models = [self._load_model(name) for name in expert_model_names]
        self.merged_model: Optional[AutoModelForCausalLM] = None
        self.safety_classifier = self._load_safety_classifier()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2').to(device)

    def _load_model(self, model_name: str) -> ModelConfig:
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_auth_token=self.hf_token).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=self.hf_token)
        return ModelConfig(model_name, model, tokenizer)

    def _load_safety_classifier(self):
        from transformers import pipeline
        return pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target", device=self.device)
    

    def generate_safety_data(self, num_samples: int = 100) -> List[Tuple[str, str]]:
        logger.info(f"Generating safety data with {num_samples} samples")
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        unsafe_queries = np.random.choice(dataset["prompt"], num_samples, replace=False)
        
        safety_data = []
        for query in tqdm(unsafe_queries, desc="Generating safety data"):
            responses = [self._generate_response(model.model, model.tokenizer, query) 
                         for model in [self.base_model] + self.expert_models]
            safe_responses = [resp for resp in responses if self._is_safe_response(resp)]
            if safe_responses:
                safety_data.append((query, np.random.choice(safe_responses)))
        
        logger.info(f"Generated {len(safety_data)} safety data points")
        return safety_data

    def generate_domain_data(self, num_samples: int = 100) -> List[Tuple[str, str, str, str]]:
        logger.info(f"Generating domain data with {num_samples} samples")
        datasets = [
            load_dataset("bigbio/med_qa", split="train"),
            load_dataset("metaeval/mathqa", split="train"),
        ]
        
        domain_data = []
        for dataset in datasets:
            for item in tqdm(np.random.choice(dataset, num_samples // len(datasets), replace=False), desc=f"Generating data for {dataset.name}"):
                question = item['question']
                answer = item.get('answer', item.get('correct'))
                for model in self.expert_models:
                    response = self._generate_response(model.model, model.tokenizer, question)
                    domain_data.append((question, response, answer, dataset.name))
        
        logger.info(f"Generated {len(domain_data)} domain data points")
        return domain_data

    @torch.no_grad()
    def _generate_response(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, query: str) -> str:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _is_safe_response(self, response: str) -> bool:
        result = self.safety_classifier(response)[0]
        return result['label'] == 'LABEL_0' and result['score'] > 0.9

    def dare_ties_merge(self, lambdas: List[float], sparsity: float = 0.5):
        logger.info(f"Performing DARE-TIES merge with lambdas: {lambdas}, sparsity: {sparsity}")
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

    def elastic_weight_consolidation(self, lambdas: List[float], fisher_importance: Dict[str, torch.Tensor]):
        logger.info("Performing Elastic Weight Consolidation")
        merged_state_dict = {}
        for param_name, base_param in self.base_model.model.named_parameters():
            merged_param = base_param * lambdas[0]
            for i, expert_model in enumerate(self.expert_models):
                expert_param = dict(expert_model.model.named_parameters())[param_name]
                merged_param += expert_param * lambdas[i+1]
            
            if param_name in fisher_importance:
                ewc_term = fisher_importance[param_name] * (merged_param - base_param) ** 2
                merged_param -= 0.5 * ewc_term  # Adjust the strength of EWC as needed
            
            merged_state_dict[param_name] = merged_param
        
        self.merged_model = AutoModelForCausalLM.from_pretrained(
            self.base_model.model.config._name_or_path,
            state_dict=merged_state_dict,
            torch_dtype=torch.float16
        ).to(self.device)

    def evaluate_model(
        self,
        model: AutoModelForCausalLM,
        safety_data: List[Tuple[str, str]],
        domain_data: List[Tuple[str, str, str, str]]
    ) -> Tuple[float, Dict[str, float], float]:
        logger.info("Starting model evaluation")
        safety_score = sum(self._is_safe_response(self._generate_response(model, self.base_model.tokenizer, query))
                           for query, _ in tqdm(safety_data, desc="Evaluating safety"))
        
        domain_scores = {dataset: 0 for dataset in set(data[3] for data in domain_data)}
        semantic_similarities = []
        for question, _, answer, dataset in tqdm(domain_data, desc="Evaluating domain expertise"):
            response = self._generate_response(model, self.base_model.tokenizer, question)
            if answer.lower() in response.lower():
                domain_scores[dataset] += 1
            
            # Calculate semantic similarity
            response_embedding = self.sentence_transformer.encode(response, convert_to_tensor=True)
            answer_embedding = self.sentence_transformer.encode(answer, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(response_embedding, answer_embedding).item()
            semantic_similarities.append(similarity)
        
        safety_score /= len(safety_data)
        domain_scores = {k: v / len([d for d in domain_data if d[3] == k]) for k, v in domain_scores.items()}
        avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities)
        
        logger.info(f"Evaluation complete. Safety score: {safety_score}, Domain scores: {domain_scores}, Avg Semantic Similarity: {avg_semantic_similarity}")
        return safety_score, domain_scores, avg_semantic_similarity

    def objective_function(
        self,
        params: List[float],
        safety_data: List[Tuple[str, str]],
        domain_data: List[Tuple[str, str, str, str]],
        merge_method: str = 'dare_ties'
    ) -> float:
        logger.info(f"Calculating objective function with params: {params}")
        lambdas, sparsity = params[:-1], params[-1]
        
        if merge_method == 'dare_ties':
            self.dare_ties_merge(lambdas, sparsity)
        elif merge_method == 'ewc':
            fisher_importance = self.compute_fisher_importance()
            self.elastic_weight_consolidation(lambdas, fisher_importance)
        else:
            raise ValueError(f"Unknown merge method: {merge_method}")
        
        safety_score, domain_scores, avg_semantic_similarity = self.evaluate_model(self.merged_model, safety_data, domain_data)
        objective_value = -(safety_score + sum(domain_scores.values()) / len(domain_scores) + avg_semantic_similarity)
        logger.info(f"Objective function value: {objective_value}")
        wandb.log({
            "safety_score": safety_score,
            "domain_scores": domain_scores,
            "avg_semantic_similarity": avg_semantic_similarity,
            "objective_value": objective_value
        })
        return objective_value

    def compute_fisher_importance(self) -> Dict[str, torch.Tensor]:
        logger.info("Computing Fisher Information")
        fisher_importance = {}
        self.base_model.model.eval()
        for param_name, param in self.base_model.model.named_parameters():
            fisher_importance[param_name] = torch.zeros_like(param)
        
        # You'd need a validation set here. For simplicity, we'll use a small synthetic dataset
        validation_data = ["This is a sample sentence.", "Another example for Fisher computation."]
        
        for sentence in validation_data:
            self.base_model.model.zero_grad()
            inputs = self.base_model.tokenizer(sentence, return_tensors="pt").to(self.device)
            outputs = self.base_model.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            
            for param_name, param in self.base_model.model.named_parameters():
                fisher_importance[param_name] += param.grad.data ** 2 / len(validation_data)
        
        return fisher_importance

    def evomm_optimize(
        self,
        safety_data: List[Tuple[str, str]],
        domain_data: List[Tuple[str, str, str, str]],
        merge_method: str = 'dare_ties'
    ) -> List[float]:
        logger.info(f"Starting EVOMM optimization with {merge_method} merge method")
        bounds = [(0, 1)] * (len(self.expert_models) + 2)  # +1 for base model, +1 for sparsity
        result = differential_evolution(
            partial(self.objective_function, safety_data=safety_data, domain_data=domain_data, merge_method=merge_method),
            bounds,
            popsize=20,
            maxiter=100,
            workers=-1  # Use all available CPU cores
        )
        logger.info(f"EVOMM optimization complete. Optimal params: {result.x}")
        return result.x

    def fine_tune(self, dataset: str, num_epochs: int = 3):
        logger.info(f"Fine-tuning merged model on {dataset} for {num_epochs} epochs")
        train_dataset = load_dataset(dataset, split="train")
        
        trainer = Trainer(
            model=self.merged_model,
            train_dataset=train_dataset,
            args=TrainingArguments(
                output_dir="./fine_tuned_model",
                num_train_epochs=num_epochs,
                per_device_train_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
            ),
        )
        
        trainer.train()
        logger.info("Fine-tuning complete")

    def save_model(self, output_dir: str):
        logger.info(f"Saving merged model to {output_dir}")
        self.merged_model.save_pretrained(output_dir)
        self.base_model.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully")

def setup_sagemaker_environment(args):
    session = boto3.Session(region_name=args.aws_region)
    sagemaker_session = sagemaker.Session(boto_session=session)
    
    role = sagemaker.get_execution_role()
    
    pytorch_estimator = PyTorch(
        entry_point='sagemaker_script.py',
        role=role,
        instance_count=1,
        instance_type=args.sagemaker_instance_type,
        framework_version='1.8.1',
        py_version='py3',
        hyperparameters={
            'base_model': args.base_model,
            'expert_models': ' '.join(args.expert_models),
            'hf_token': args.hf_token,
            'num_samples': args.num_samples,
            'merge_method': args.merge_method,
            'fine_tune': args.fine_tune,
            'fine_tune_dataset': args.fine_tune_dataset,
            'fine_tune_epochs': args.fine_tune_epochs,
            'save_model': args.save_model,
            'output_dir': args.output_dir
        }
    )
    
    pytorch_estimator.fit()
    
    logger.info("SageMaker training job completed")

def parse_args():
    parser = argparse.ArgumentParser(description="Safety-Aware Model Merger")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--expert_models", nargs="+", default=["microsoft/BioGPT-Large", "EleutherAI/gpt-neox-20b"], help="Expert model names")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples for safety and domain data")
    parser.add_argument("--merge_method", type=str, choices=['dare_ties', 'ewc'], default='dare_ties', help="Merge method to use")
    parser.add_argument("--fine_tune", action="store_true", help="Whether to fine-tune the merged model")
    parser.add_argument("--fine_tune_dataset", type=str, default="wikipedia", help="Dataset to use for fine-tuning")
    parser.add_argument("--fine_tune_epochs", type=int, default=3, help="Number of epochs for fine-tuning")
    parser.add_argument("--save_model", action="store_true", help="Whether to save the merged model")
    parser.add_argument("--output_dir", type=str, default="./merged_model", help="Directory to save the merged model")
    parser.add_argument("--use_sagemaker", action="store_true", help="Whether to use Amazon SageMaker for computation")
    parser.add_argument("--aws_region", type=str, default="us-west-2", help="AWS region")
    parser.add_argument("--sagemaker_instance_type", type=str, default="ml.p3.2xlarge", help="SageMaker instance type")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.use_sagemaker:
        logger.info("Setting up SageMaker environment")
        setup_sagemaker_environment(args)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        merger = SafetyAwareMerger(args.base_model, args.expert_models, device, args.hf_token)
        
        safety_data = merger.generate_safety_data(num_samples=args.num_samples)
        domain_data = merger.generate_domain_data(num_samples=args.num_samples)
        
        optimal_params = merger.evomm_optimize(safety_data, domain_data, merge_method=args.merge_method)
        
        if args.merge_method == 'dare_ties':
            merger.dare_ties_merge(optimal_params[:-1], optimal_params[-1])
        elif args.merge_method == 'ewc':
            fisher_importance = merger.compute_fisher_importance()
            merger.elastic_weight_consolidation(optimal_params[:-1], fisher_importance)
        
        if args.fine_tune:
            merger.fine_tune(args.fine_tune_dataset, args.fine_tune_epochs)
        
        safety_score, domain_scores, avg_semantic_similarity = merger.evaluate_model(merger.merged_model, safety_data, domain_data)
        logger.info(f"Final Merged Model - Safety: {safety_score:.2f}, Domain Performances: {domain_scores}, Avg Semantic Similarity: {avg_semantic_similarity}")
        logger.info(f"Optimal parameters: {optimal_params}")
        
        if args.save_model:
            merger.save_model(args.output_dir)

if __name__ == "__main__":
    main()