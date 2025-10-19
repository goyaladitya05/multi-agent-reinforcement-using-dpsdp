import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
import torch

cache_path = "/data/adityagoyal/cache"
os.environ['HF_HOME'] = cache_path

dpo_dataset = load_dataset("json", data_files="dpo_dataset.jsonl", split="train")

actor_model_path = "./sft_actor_express"
critic_model_path = "./sft_critic_express"
tokenizer = AutoTokenizer.from_pretrained(actor_model_path)

print("--- Training DPO Actor ---")
actor_model = AutoModelForCausalLM.from_pretrained(
    actor_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

actor_dpo_args = TrainingArguments(
    output_dir="./dpo_actor_express",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=2e-7,
    max_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=10,
    bf16=True,
)

actor_dpo_trainer = DPOTrainer(
    model=actor_model,
    ref_model=None,
    args=actor_dpo_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    beta=0.1,
)

actor_dpo_trainer.train()
actor_dpo_trainer.save_model("./dpo_actor_express")
print("✅ DPO Actor Saved.")

print("--- Training DPO Critic ---")
critic_model = AutoModelForCausalLM.from_pretrained(
    critic_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

critic_dpo_args = TrainingArguments(
    output_dir="./dpo_critic_express",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=2e-7,
    max_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=10,
    bf16=True,
)

critic_dpo_trainer = DPOTrainer(
    model=critic_model,
    ref_model=None,
    args=critic_dpo_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    beta=0.1,
)

critic_dpo_trainer.train()
critic_dpo_trainer.save_model("./dpo_critic_express")
print("DPO Critic Saved.")