# LLaMA 3.1 QA Fine-Tune

This repository provides tools and scripts for fine-tuning the LLaMA 3.1 model using a custom question-answer dataset. The project leverages Hugging Face’s Transformers library, `PEFT`, `LoRA`, and `Unsloth` for optimized fine-tuning and inference.

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/etuckerman/LLaMA-3.1-QA-Finetune
    cd LLaMA-3.1-QA-Finetune
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    ```

3. Install `xformers` based on your Torch version:

    ```python
    from torch import __version__; from packaging.version import Version as V
    xformers = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"
    !pip install --no-deps {xformers} trl peft accelerate bitsandbytes triton
    ```

## Dataset

The default dataset, `qa_examples.csv`, contains RV-related question-answer pairs. You can replace this with any custom question-answer dataset in `.csv` format, ensuring it has two columns: `question` and `answer`.

## Fine-Tuning

To fine-tune the LLaMA 3.1 model with your dataset, adjust the configuration files as needed and run:

```bash
torchrun --nproc_per_node=2 train.py
