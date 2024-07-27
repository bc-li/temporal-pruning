# temporal-pruning

This repository provides the implementation for paper "Few-shot Temporal Pruning Accelerates Diffusion Models for Text Generation" (LREC-COLING 2024). 

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
