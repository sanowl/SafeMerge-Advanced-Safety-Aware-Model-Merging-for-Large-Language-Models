


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
git clone https://github.com/sanowl/SafeMerge-Advanced-Safety-Aware-Model-Merging-for-Large-Language-Models.git
cd SafeMerge-Advanced-Safety-Aware-Model-Merging-for-Large-Language-Models
pip install -r requirements.txt
```

## Usage

```python
from safemerge import SafetyAwareMerger

merger = SafetyAwareMerger(
    "mistralai/Mistral-7B-Instruct-v0.2",
    ["microsoft/BioGPT-Large", "EleutherAI/gpt-neox-20b"],
    device
)

safety_data = merger.generate_safety_data(num_samples=1000)
domain_data = merger.generate_domain_data(num_samples=1000)

optimal_params = merger.evomm_optimize(safety_data, domain_data)
merger.dare_ties_merge(optimal_params[:-1], optimal_params[-1])

safety_score, domain_scores = merger.evaluate_model(merger.merged_model, safety_data, domain_data)
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- CUDA-capable GPU(s) with 16GB+ memory

## Citing

If you use SafeMerge in your research, please cite the original paper:

```
@article{hammoud2024model,
  title={Model Merging and Safety Alignment: One Bad Model Spoils the Bunch},
  author={Hammoud, Hasan Abed Al Kader and Michieli, Umberto and Pizzati, Fabio and Torr, Philip and Bibi, Adel and Ghanem, Bernard and Ozay, Mete},
  journal={arXiv preprint arXiv:2406.14563},
  year={2024}
}
```

## Contributing

We welcome contributions to SafeMerge! Please feel free to submit pull requests, report issues, or request features through the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The authors of the original research paper for their groundbreaking work in safety-aware model merging
- The open-source community for providing essential tools and libraries

## Disclaimer

SafeMerge is an implementation of advanced AI techniques and should be used responsibly. While it aims to improve safety in merged models, it does not guarantee complete safety or alignment. Users should always carefully evaluate the output of AI models, especially in sensitive applications.

