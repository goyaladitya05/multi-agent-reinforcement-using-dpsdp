# **Multi-Agent Reinforcement Learning for LLM Reasoning**

This repository contains the implementation of a multi-agent reinforcement learning framework designed to enhance the mathematical reasoning capabilities of Large Language Models (LLMs). The project follows the methodology outlined in the research paper "Reinforce LLM Reasoning through Multi-Agent Reflection," modeling the problem as a Markov Decision Process (MDP) and utilizing Direct Preference Optimization (DPO) for policy alignment.

The entire pipeline, from data generation to final model evaluation, was executed on high-performance computing hardware.

---

## **Hardware Used**

-   **GPU**: NVIDIA A100 80GB
-   **Role**: The A100 was instrumental for all computationally intensive tasks, including:
    -   Large-scale inference with 8B and 70B parameter models using the vLLM engine.
    -   Quantized loading (`AWQ`) of the 70B oracle model to fit within VRAM constraints.
    -   Accelerated model training (SFT and DPO) using `bfloat16` mixed-precision and Flash Attention 2.

---

## **Project Pipeline**

The project was executed in three distinct phases: initial supervised fine-tuning to teach the models their roles, reinforcement learning to refine their behavior, and a final evaluation to measure performance.

### **1. Supervised Fine-Tuning (SFT)**

The goal of this phase was to provide the base 8B Llama model with the fundamental skills required for its roles as an "actor" (solver) and a "critic" (reviewer).

-   **Data Generation**: A high-quality "oracle" dataset was created. For each problem in a subset of the `OpenMathInstruct-2` dataset, a powerful 70B Llama model was used to generate critical feedback and a corrected, refined solution for an initial answer provided by the 8B model.
-   **SFT Training**: Two specialized models were trained on this oracle data:
    -   **SFT Actor**: Learned to generate a refined solution given a problem, an initial answer, and feedback.
    -   **SFT Critic**: Learned to generate constructive feedback given a problem and an initial answer.

### **2. Reinforcement Learning via Direct Preference Optimization (DPO)**

This phase used reinforcement learning to improve the SFT models by teaching them to prefer correct reasoning paths over incorrect ones.

-   **Preference Data Generation**: A preference dataset was created instantly using a "zero-inference" shortcut. By comparing the `initial_answer` and `refined_answer` from the SFT dataset to the ground truth, we generated a collection of (`prompt`, `chosen`, `rejected`) triplets.
-   **DPO Training**: The `DPOTrainer` from the TRL library was used to fine-tune the SFT models on this preference data. This process aligns the models' policies with the desired behavior, making the actor a more accurate solver and the critic a more effective guide.

### **3. Evaluation**

The final DPO-trained models were evaluated on the **MATH 500 benchmark** to measure the performance uplift from the multi-agent refinement process. The key metrics were:

-   **`pass1@turn1` (p1@t1)**: The accuracy of the actor's first, unassisted attempt.
-   **`maj1@turn5` (m1@t5)**: The accuracy of the majority vote answer after five rounds of critique and refinement.
-   **`pass1@turn5` (p1@t5)**: The success rate of finding at least one correct answer within the five attempts.

---

## **Final Results**

![Final Evaluation Metrics](output.jpg)

---

### **Note on Scale**

This project is a smaller-scale implementation of the methodology described in the research paper **"Reinforce LLM Reasoning through Multi-Agent Reflection"**. It was designed to build and validate a working end-to-end pipeline, from data generation to final evaluation, demonstrating the core concepts of multi-agent collaboration and reinforcement learning for LLM reasoning.