
import os

cache_path = "/data/adityagoyal/cache"
os.environ['HF_HOME'] = cache_path
os.environ['VLLM_CACHE_HOME'] = cache_path

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
import torch
from datasets import load_dataset

actor_model_path = "./sft_actor"

#loading DPO preference dataset
dpo_dataset = load_dataset("json", data_files="dpo_dataset.jsonl", split="train")

actor_dpo_data = dpo_dataset.filter(lambda x: x['turn'] in [0, 2])
critic_dpo_data = dpo_dataset.filter(lambda x: x['turn'] == 1)

tokenizer = AutoTokenizer.from_pretrained(actor_model_path)

print("Training DPO Actor")
actor_model = AutoModelForCausalLM.from_pretrained(actor_model_path)

dpo_args = TrainingArguments(
    output_dir="./dpo_actor",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=2e-7,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=10,
    bf16=True,
)

actor_dpo_trainer = DPOTrainer(
    model=actor_model,
    ref_model=None,
    args=dpo_args,
    train_dataset=actor_dpo_data,
    tokenizer=tokenizer,
    beta=0.1,
)

actor_dpo_trainer.train()
actor_dpo_trainer.save_model("./dpo_actor")
print("DPO Actor Saved.")

print("--- Training DPO Critic ---")

# Load the SFT-trained critic model
critic_model_path = "./sft_critic"
critic_model = AutoModelForCausalLM.from_pretrained(
    critic_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)


critic_dpo_args = TrainingArguments(
    output_dir="./dpo_critic",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=2e-7,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=10,
    bf16=True,
)


critic_dpo_trainer = DPOTrainer(
    model=critic_model,
    ref_model=None,
    args=critic_dpo_args,
    train_dataset=critic_dpo_data,
    tokenizer=tokenizer,
    beta=0.1,  # KL regularization parameter
)


critic_dpo_trainer.train()

critic_dpo_trainer.save_model("./dpo_critic")
print("DPO Critic Saved.")