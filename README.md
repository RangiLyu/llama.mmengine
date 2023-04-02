# LLaMA.MMEngine

**😋Training LLaMA with MMEngine!**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/Code%20License-Apache_2.0-purple.svg)](./LICENSE)



**LLaMA.MMEngine** is an experimental repository that leverages the [MMEngine](https://github.com/open-mmlab/mmengine) training engine, originally designed for computer vision tasks, to train and fine-tune language models. The primary goal of this project is to explore the compatibility of MMEngine with language models, learn about fine-tuning techniques, and engage with the open-source community for knowledge sharing and collaboration.

## 🤩 Features

- Support for loading LLaMA models with parameter sizes ranging from 7B to 65B
- Instruct tuning support
- low-rank adaptation (LoRA) fine-tuning support

## 🏃 Todo-List

- [ ] int8 quantization support
- [ ] improve the generate script
- [ ] support show validation loss

## 👀 Getting Started

### Installation

1. Install PyTorch
   
    Following this guide https://pytorch.org/get-started/locally/

2. Setup this repo

    Clone the repo
    ```shell
    git clone https://github.com/RangiLyu/llama.mmengine
    cd llama.mmengine
    ```
    Install dependencies
    ```shell
    pip install -r requirements.txt
    ```
    Run setup.py
    ```shell
    python setup.py develop
    ```

### Get pre-trained LLaMA models

Please Download the model weights from the [official LLaMA repo](https://github.com/facebookresearch/llama).

The checkpoints folder should be like this:

```
checkpoints/llama
├── 7B
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
├── 13B
│   ...
├── tokenizer_checklist.chk
└── tokenizer.model
```

Convert the weights (Thanks for the script from [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama)):

```
python scripts/convert_checkpoint.py \
    --output_dir checkpoints/mm-llama \
    --ckpt_dir checkpoints/llama \
    --tokenizer_path checkpoints/llama/tokenizer.model \
    --model_size 7B
```


### LoRA fine-tuning

```shell
python tools/train.py configs/llama-7B_finetune_3e.py
```

### Inference

```shell
python tools/generate.py configs/llama-7B_finetune_3e.py work_dirs/llama-7B_finetune_3e/epoch_3.pth
```

## 🤗 Contributing

I greatly appreciate your interest in contributing to LLaMA.MMEngine! Please note that this project is maintained as a personal side project, which means that available time for development and support is limited. With that in mind, I kindly encourage members of the community to get involved and actively contribute by submitting pull requests!

## Acknowledgements

- @Lightning-AI for [Lit-LLaMA ️](https://github.com/Lightning-AI/lit-llama)
- @tloen for [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [LLaMA](https://github.com/facebookresearch/llama)


