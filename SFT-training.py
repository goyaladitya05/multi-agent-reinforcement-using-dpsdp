import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch

cache_path = "/data/adityagoyal/cache"
os.environ['HF_HOME'] = cache_path
os.environ['VLLM_CACHE_HOME'] = cache_path

dataset = load_dataset("json", data_files="sft_dataset.jsonl", split="train")


response_template_actor = "\n\nRefined Answer:\n"
def format_actor_prompt(sample):
    return f"Problem:\n{sample['problem']}\n\nInitial Answer:\n{sample['initial_answer']}\n\nFeedback:\n{sample['feedback']}{response_template_actor}{sample['refined_answer']}"

response_template_critic = "\n\nFeedback:\n"
def format_critic_prompt(sample):
    return f"Problem:\n{sample['problem']}\n\nInitial Answer:\n{sample['initial_answer']}{response_template_critic}{sample['feedback']}"


model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)
tokenizer.pad_token = tokenizer.eos_token


print("--- Training SFT Actor ---")
actor_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    cache_dir=cache_path
)

data_collator_actor = DataCollatorForCompletionOnlyLM(response_template=response_template_actor, tokenizer=tokenizer)

actor_training_args = TrainingArguments(
    output_dir="./sft_actor_express",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=1e-6,
    max_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=10,
    bf16=True,
)

actor_trainer = SFTTrainer(
    model=actor_model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=format_actor_prompt,
    data_collator=data_collator_actor,
    args=actor_training_args,
    max_seq_length=4096,
)

actor_trainer.train()
actor_trainer.save_model("./sft_actor_express")
print("✅ Actor Model Saved.")

print("--- Training SFT Critic ---")
critic_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    cache_dir=cache_path
)

data_collator_critic = DataCollatorForCompletionOnlyLM(response_template=response_template_critic, tokenizer=tokenizer)

critic_training_args = TrainingArguments(
    output_dir="./sft_critic_express",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=1e-6,
    max_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=10,
    bf16=True,
)

critic_trainer = SFTTrainer(
    model=critic_model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=format_critic_prompt,
    data_collator=data_collator_critic,
    args=critic_training_args,
    max_seq_length=4096,
)

critic_trainer.train()
critic_trainer.save_model("./sft_critic_express")
print("Critic model saved.")