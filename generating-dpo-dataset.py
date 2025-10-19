import os
import pandas as pd
from datasets import load_dataset
import re

cache_path = "/data/adityagoyal/cache"
os.environ['HF_HOME'] = cache_path

print("Loading datasets...")
sft_df = pd.read_json("sft_dataset.jsonl", lines=True)
original_dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train", cache_dir=cache_path).select(range(len(sft_df)))

dpo_data = []

def is_correct(model_output, ground_truth_answer):
    try:
        matches = re.findall(r'\\boxed\{(.+?)\}', str(model_output))
        if not matches:
            return False
        model_answer = matches[-1]
        model_answer_clean = str(model_answer).replace(",", "").strip()
        ground_truth_clean = str(ground_truth_answer).replace(",", "").strip()
        return model_answer_clean == ground_truth_clean
    except Exception:
        return False

print("Generating preference pairs...")
for index, row in sft_df.iterrows():
    ground_truth_answer = original_dataset[index]['expected_answer']

    initial_was_correct = is_correct(row['initial_answer'], ground_truth_answer)
    refined_was_correct = is_correct(row['refined_answer'], ground_truth_answer)

    if not initial_was_correct and refined_was_correct:
        dpo_data.append({
            "prompt": f"Problem:\n{row['problem']}",
            "chosen": row['refined_answer'],
            "rejected": row['initial_answer']
        })

dpo_df = pd.DataFrame(dpo_data)
dpo_df.to_json("dpo_dataset.jsonl", orient="records", lines=True)

print(f"Generated {len(dpo_df)} preference pairs instantly from {len(sft_df)} SFT samples.")