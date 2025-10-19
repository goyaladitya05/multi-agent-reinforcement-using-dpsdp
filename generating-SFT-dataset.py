import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
import os
import torch
import gc

os.environ['HF_HOME'] = '/data/adityagoyal/cache'

base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
oracle_model_id = "meta-llama/Llama-3.1-70B-Instruct"

dataset=load_dataset("nvidia/OpenMathInstruct-2",split="train")
dataset_subset=dataset.select(range(100))


prompt_initial_answer="""You are an AI language model designed to assist with math problem-solving. In this task, I will provide you with math problems. Your goal is to solve the problem step-by-step, showing your reasoning at each step. After you have finished solving the problem, present your final answer as \\boxed{Your Answer}.
{problem}"""

prompt_feedback = """Take a moment to review your previous response for accuracy and completeness. Briefly identify any errors or gaps that might lead to incorrect answers, but don't worry about fixing them just focus on pointing them out."""

prompt_refined_answer = """Using your reflection as a guide, carefully make any necessary corrections to your previous response. Ensure your final answer addresses any identified issues and present it as \\boxed{Your Answer}. Pay close attention to the errors you pinpointed during the reflection."""

print("Loading Base Model..")
base_model = LLM(
    model=base_model_id,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6
)

print("Base Model Loaded Successfully.")
print("Generating Initial answers with Base Model..")

results_data = []
for item in dataset_subset:
    problem_text = item['question']
    sampling_params_base = SamplingParams(temperature=0.8, max_tokens=1024)
    initial_prompt = prompt_initial_answer.format(problem=problem_text)
    outputs = base_model.generate(initial_prompt, sampling_params=sampling_params_base)
    initial_answer = outputs[0].outputs[0].text
    results_data.append({'problem': problem_text, 'initial_answer': initial_answer})

print("Initial answers generated.")

#unloading base model
del base_model
gc.collect()
torch.cuda.empty_cache()
print("Base Model Unloaded")

print("Loading Oracle Model..")

oracle_model = LLM(
    model=oracle_model_id,
    quantization='awq',
    dtype='half',
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.6
)

final_results = []
for item in results_data:
    problem_text = item['problem']
    initial_answer = item['initial_answer']

    # Generate feedback
    sampling_params_oracle = SamplingParams(temperature=0.0, max_tokens=512)
    feedback_context = f"Problem:\n{problem_text}\n\nInitial Answer:\n{initial_answer}\n\nFeedback Request:\n{prompt_feedback}"
    outputs = oracle_model.generate(feedback_context, sampling_params_oracle)
    feedback = outputs[0].outputs[0].text

    # Generate refined answer
    refined_context = f"Problem:\n{problem_text}\n\nInitial Answer:\n{initial_answer}\n\nFeedback:\n{feedback}\n\nRefinement Request:\n{prompt_refined_answer}"
    outputs = oracle_model.generate(refined_context, sampling_params_oracle)
    refined_answer = outputs[0].outputs[0].text

    final_results.append({
        'problem': problem_text,
        'initial_answer': initial_answer,
        'feedback': feedback,
        'refined_answer': refined_answer
    })

print("Feedback and refined answers generated.")

# --- Unloading the final model ---
del oracle_model
gc.collect()
torch.cuda.empty_cache()
print("Oracle Model Unloaded")


df=pd.DataFrame(final_results)
df.to_json("sft_dataset.json",orient="records",lines=True)
print("Dataset Saved Successfully.")