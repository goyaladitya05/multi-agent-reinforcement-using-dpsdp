import os
import torch
import gc
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams
from collections import Counter
import pandas as pd

cache_path = "/data/adityagoyal/cache"
os.environ['HF_HOME'] = cache_path
os.environ['VLLM_CACHE_HOME'] = cache_path

print("--- Loading DPO Models ---")
actor_model_path = "./dpo_actor_express"
critic_model_path = "./dpo_critic_express"

actor_model = LLM(model=actor_model_path, gpu_memory_utilization=0.6, download_dir=cache_path)
critic_model = LLM(model=critic_model_path, gpu_memory_utilization=0.6, download_dir=cache_path)

print("--- Loading MATH 500 Benchmark ---")
benchmark_dataset = load_dataset("hendrycks/math", "test", cache_dir=cache_path).select(range(500))

prompt_initial_answer="""You are an AI language model designed to assist with math problem-solving. In this task, I will provide you with math problems. Your goal is to solve the problem step-by-step, showing your reasoning at each step. After you have finished solving the problem, present your final answer as \\boxed{{Your Answer}}.
{problem}"""
prompt_feedback = """Take a moment to review your previous response for accuracy and completeness. Briefly identify any errors or gaps that might lead to incorrect answers, but don't worry about fixing them just focus on pointing them out."""
prompt_refined_answer = """Using your reflection as a guide, carefully make any necessary corrections to your previous response. Ensure your final answer addresses any identified issues and present it as \\boxed{{Your Answer}}. Pay close attention to the errors you pinpointed during the reflection."""

def is_correct(model_output, ground_truth_answer):
    try:
        matches = re.findall(r'\\boxed\{(.+?)\}', str(model_output))
        if not matches: return False
        model_answer_clean = str(matches[-1]).replace(",", "").strip()
        ground_truth_clean = str(ground_truth_answer).replace(",", "").strip()
        return model_answer_clean == ground_truth_clean
    except Exception:
        return False

print("--- Starting Evaluation ---")
eval_params = SamplingParams(temperature=0.0, max_tokens=2048)
all_results = []

for item in benchmark_dataset:
    problem_text = item['problem']
    ground_truth = item['solution']
    actor_answers = []

    #Initial Answer
    initial_prompt = prompt_initial_answer.format(problem=problem_text)
    outputs = actor_model.generate(initial_prompt, eval_params)
    current_answer = outputs[0].outputs[0].text
    actor_answers.append(current_answer)

    #Refinement Loop
    for _ in range(4):
        feedback_context = f"Problem:\n{problem_text}\n\nInitial Answer:\n{current_answer}\n\nFeedback Request:\n{prompt_feedback}"
        outputs = critic_model.generate(feedback_context, eval_params)
        feedback = outputs[0].outputs[0].text

        refined_context = f"Problem:\n{problem_text}\n\nInitial Answer:\n{current_answer}\n\nFeedback:\n{feedback}\n\nRefinement Request:\n{prompt_refined_answer}"
        outputs = actor_model.generate(refined_context, eval_params)
        current_answer = outputs[0].outputs[0].text
        actor_answers.append(current_answer)

    all_results.append({
        'problem': problem_text,
        'ground_truth': ground_truth,
        'generated_answers': actor_answers
    })

print("--- Evaluation Complete ---")


p1_at_t1_correct = 0
maj1_at_t5_correct = 0
p1_at_t5_correct = 0
total = len(all_results)

for result in all_results:
    ground_truth = result['ground_truth']
    answers = result['generated_answers']

    if is_correct(answers[0], ground_truth):
        p1_at_t1_correct += 1

    if any(is_correct(ans, ground_truth) for ans in answers):
        p1_at_t5_correct += 1

    answer_counts = Counter(re.findall(r'\\boxed\{(.+?)\}', str(ans))[-1] for ans in answers if re.findall(r'\\boxed\{(.+?)\}', str(ans)))
    if answer_counts:
        majority_answer = answer_counts.most_common(1)[0][0]
        if is_correct(f"\\boxed{{{majority_answer}}}", ground_truth):
            maj1_at_t5_correct += 1

print("\n--- Final Results ---")
print(f"Total Problems: {total}")
print(f"pass1@turn1 (p1@t1): {p1_at_t1_correct / total:.4f}")
print(f"maj1@turn5 (m1@t5): {maj1_at_t5_correct / total:.4f}")
print(f"pass1@turn5 (p1@t5): {p1_at_t5_correct / total:.4f}")

del actor_model, critic_model
gc.collect()
torch.cuda.empty_cache()