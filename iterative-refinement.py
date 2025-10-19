import torch
import gc
from vllm import LLM, SamplingParams

# --- 1. Load your final DPO-trained models ---
print("--- Loading DPO Models ---")
actor_model = LLM(model="./dpo_actor")
critic_model = LLM(model="./dpo_critic")


# ... (code to load your benchmark data) ...


# The paper uses a temperature of 0 for deterministic evaluation
eval_params = SamplingParams(temperature=0.0, max_tokens=1024)

all_results = []
for problem in benchmark_dataset:
    conversation_history = []
    actor_answers = []

    initial_prompt = format_initial_prompt(problem['problem'])  # Use a formatted prompt
    outputs = actor_model.generate(initial_prompt, eval_params)
    current_answer = outputs[0].outputs[0].text
    actor_answers.append(current_answer)
    conversation_history.append(current_answer)

    for i in range(4):
        # Critic provides feedback
        feedback_prompt = format_feedback_prompt(problem['problem'], conversation_history)
        outputs = critic_model.generate(feedback_prompt, eval_params)
        feedback = outputs[0].outputs[0].text
        conversation_history.append(feedback)

        # Actor refines answer based on feedback
        refinement_prompt = format_refinement_prompt(problem['problem'], conversation_history)
        outputs = actor_model.generate(refinement_prompt, eval_params)
        current_answer = outputs[0].outputs[0].text
        actor_answers.append(current_answer)
        conversation_history.append(current_answer)

    all_results.append({
        'problem': problem['problem'],
        'ground_truth': problem['expected_answer'],
        'generated_answers': actor_answers
    })

del actor_model
del critic_model
gc.collect()
torch.cuda.empty_cache()
