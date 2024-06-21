# SafeMerge: Advanced Safety-Aware Model Merging for Large Language Models

## Overview

SafeMerge is an advanced implementation of safety-aware model merging for Large Language Models (LLMs), based on the research paper "Model Merging and Safety Alignment: One Bad Model Spoils the Bunch" by Hasan Abed Al Kader Hammoud et al. (2024).

This project addresses the critical issue of maintaining safety alignment when merging multiple expert LLMs into a single, versatile model. It implements a sophisticated approach to combine the strengths of various models while preserving and optimizing their safety characteristics.

## Key Features

- Implementation of DARE-TIES merging technique with adjustable sparsity
- EvoMM optimization using differential evolution
- Multi-GPU support with PyTorch's DistributedDataParallel (DDP)
- Advanced safety evaluation using pre-trained RoBERTa model
- Support for multiple expert models and domain-specific datasets
- Efficient data generation for both safety and domain expertise

## Installation

```bash
[git clone https://github.com/sanowl/SafeMerge.git](https://github.com/sanowl/SafeMerge-Advanced-Safety-Aware-Model-Merging-for-Large-Language-Models.git)
cd SafeMerge
pip install -r requirements.txt
