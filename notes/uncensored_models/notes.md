# Uncensored Models

- [Reference](https://erichartford.com/uncensored-models)

- Most models available today are *aligned*, which stop them from doing potentially bad things such as teaching people how to cook meth or build bombs.
  - Examples of aligned models are Alpaca, Vicuna, WizardLM, MPT, etc.

## How models are aligned

- In the standard process, we start with a base model such as Pythia, and then fine tune it with an *instruction dataset*.
  - The instruction dataset is usually generated from dialogue with GPT4 (ChatGPT API), which was aligned by OpenAI. When GPT4 refuses to answer a question or answer with bias, the alignment gets passed down into the dataset and the models trained using that dataset.
  - Since the dataset will contain refusals, our model will learn what, when, and how to refuse. Ergo, our model gets aligned with GPT4's alignment. 
- So to create uncensored models, we omit examples in the instruction dataset which contains refusals and biased answers, and then train the model with the filtered dataset the same way as we would normally.

## Fine-tuning Llama 7B 

- Requires node with 4x A100 80GB and 2TB of storage. 

- Download dataset and base model 
```bash
mkdir /workspace/models
mkdir /workspace/datasets
cd /workspace/datasets
git lfs install
git clone https://huggingface.co/datasets/ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered
cd /workspace/models
git clone https://huggingface.co/huggyllama/llama-7b
cd /workspace
```

- Get code to fine tune WizardLM
```bash
conda create -n llamax python=3.10
conda activate llamax
git clone https://github.com/AetherCortex/Llama-X.git
cd Llama-X/src
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd ../..
pip install -r requirements.txt
cd src
wget https://github.com/nlpxucan/WizardLM/raw/main/src/train_freeform.py
wget https://github.com/nlpxucan/WizardLM/raw/main/src/inference_wizardlm.py
wget https://github.com/nlpxucan/WizardLM/raw/main/src/weight_diff_wizard.py
```
- Remove CPU offload from `deepspeed_config.json`
- There's a bug when model is saved, do not delete checkpoints.
```bash
deepspeed train_freeform.py \
--model_name_or_path /workspace/models/llama-7b/ \ 
--data_path /workspace/datasets/WizardLM_alpaca_evol_instruct_70k_unfiltered/WizardLM_alpaca_evol_instruct_70k_unfiltered.json \
--output_dir /workspace/models/WizardLM-7B-Uncensored/ \
--num_train_epochs 3 \
--model_max_length 2048 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 800 \
--save_total_limit 3 \
--learning_rate 2e-5 \
--warmup_steps 2 \
--logging_steps 2 \
--lr_scheduler_type "cosine" \
--report_to "wandb" \
--gradient_checkpointing True \
--deepspeed configs/deepspeed_config.json \
--fp16 True
```
- Edit `train_freeform.py` to resume from latest checkpoint, run again with lower `save_steps` (80). Then do it again with `save_steps=1`

