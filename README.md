# temporal-pruning

This repository provides the implementation for paper "[Few-shot Temporal Pruning Accelerates Diffusion Models for Text Generation](https://aclanthology.org/2024.lrec-main.637)" (LREC-COLING 2024). 

*Straightforward Bayesian optimization to accelerate Multinomial Diffusion, Absorbing Diffusion, and DiffuSeq by up to 400x.*


## Getting Started

### Discrete Diffusion Models

Our approach builds on the work from [reparam-discrete-diffusion](https://github.com/HKUNLP/reparam-discrete-diffusion), with minor modifications to suit our specific needs.

#### Setup

##### Prerequisites

Ensure you have the required dependencies installed. Follow these steps to set up your environment:

```bash
cd reparam-discrete-diffusion

pip install -r requirements.txt

# install our package of discrete diffusion models
pip install -e discrete_diffusion

# install our fork of fairseq
cd fairseq
python3 setup.py build develop
cd ..
```

##### Model Checkpoints

Download the [trained model checkpoints](https://github.com/HKUNLP/reparam-discrete-diffusion?tab=readme-ov-file#trained-model-checkpoints) from the original repository and place them in the `/models` directory.

##### Data Processing

For data processing, follow the procedures outlined in the original repository:

- [Machine Translation](https://github.com/HKUNLP/reparam-discrete-diffusion?tab=readme-ov-file#machine-translation)
- [Question Generation and Paraphrasing Tasks](https://github.com/HKUNLP/reparam-discrete-diffusion?tab=readme-ov-file#data-preprocessing-1)

For *few-shot* settings, randomly sample a subset from the validation set before binarizing your data.


Then download their [trained model checkpoints](https://github.com/HKUNLP/reparam-discrete-diffusion?tab=readme-ov-file#trained-model-checkpoints) and place them to `/models` folder.


#### Running the Model

Once you have prepared your models and datasets, you can execute the following command to perform few-shot temporal pruning:

```bash
# This command runs few-shot temporal pruning on the IWSLT task for multinomial diffusion 
# with <timesteps> sampling steps on the specified <cuda-device>.
python few_shot_discrete.py --timesteps <timesteps> --cuda_device <cuda-device> --task iwslt --run_script mt --model_path path/to/your/iwslt_multinomial_checkpoints_default_checkpoint.avg5.pt
```

#### Output

The script will generate a log file in the same directory, detailing the current steps and corresponding BLEU scores, which serve as the optimization target.


## Citation

Please cite our paper if you find this work useful:

```bibtex
@inproceedings{li-etal-2024-shot-temporal,
    title = "Few-shot Temporal Pruning Accelerates Diffusion Models for Text Generation",
    author = "Li, Bocheng and Gao, Zhujin and Zhu, Yongxin and Yin, Kun and Cao, Haoyu and Jiang, Deqiang and Xu, Linli",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    year = "2024",
    url = "https://aclanthology.org/2024.lrec-main.637",
    pages = "7259--7269",
}
```

## Acknowledgements

This project builds upon the following repositories:

- [HKUNLP/reparam-discrete-diffusion](https://github.com/HKUNLP/reparam-discrete-diffusion?tab=readme-ov-file#data-preprocessing-1)
- [Shark-NLP/DiffuSeq](https://github.com/Shark-NLP/DiffuSeq)
- [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)

We are grateful to the contributors of these repositories for their significant work and dedication.