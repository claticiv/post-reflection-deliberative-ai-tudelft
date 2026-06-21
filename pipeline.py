# %%
!pip install openai
# %%
from openai import OpenAI
import os
import json
from datetime import datetime
# %%
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)
# %%

MODEL_GENERATORS = ["llama-3.2-3b-instruct", "google/gemma-4-e2b", "qwen/qwen3-vl-4b"]
MODEL_EVALUATORS = ["llama-3.2-3b-instruct", "google/gemma-4-e2b", "qwen/qwen3-vl-4b"]

import re

# Sanitized the model names when creating the nest folder structure.
def safe_id(text):
    return re.sub(r"[^\w-]+", "", text)

TEMPERATURE = 0.3
TOP_P = 0.95
TOP_K = 40
MIN_P = 0.05
REPEAT_PENALTY = 1.1

SEEDS = list(range(42, 52))
TRANSCRIPTS = list(range(1, 9))

MAX_TURNS = 4

TERMINATE_EARLY = False

VALUE_DEFINITIONS = """
Safety is the condition of being protected from harm (or other non-desirable outcomes) caused by non-intentional failure of technical, human or organisational factors. Privacy safeguards the spontaneous, independent, and uniquely individual aspects of the self. Autonomy refers to the ability of persons to create their own identity and in this way to define themselves. Societal well-being refers to how an individual feels accepted or welcome in a society or community.
"""

# %%
transcript = []

with open("prompts/generator_prompt.txt", "r", encoding="utf-8") as f:
    generator_prompt_template = f.read()

with open("prompts/evaluator_prompt.txt", "r", encoding="utf-8") as f:
    evaluator_prompt_template = f.read()

with open("prompts/generator_prompt_multi.txt", "r", encoding="utf-8") as f:
    generator_prompt_template_multi = f.read()

with open("prompts/evaluator_prompt_multi.txt", "r", encoding="utf-8") as f:
    evaluator_prompt_template_multi = f.read()

with open("prompts/evaluator_prompt_finish.txt", "r", encoding="utf-8") as f:
    evaluator_prompt_template_multi_final = f.read()

with open("anonymised_transcripts/transcript_01.txt", "r", encoding="utf-8") as f:
    transcript = f.read()

# %%
def run_transcript(transcript_id):

    transcript_path = f"anonymised_transcripts/transcript_{transcript_id:02d}.txt"

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    
    return transcript

# %%
def call_llm(messages, model, seed):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=seed,
        extra_body={
            "top_k": TOP_K,
            "min_p": MIN_P,
            "repeat_penalty": REPEAT_PENALTY
        }
    ).choices[0].message.content

# %%
def run_single_turn(transcript_id, seed, generator, evaluator):

    transcript = run_transcript(transcript_id)
    current_generator_prompt = generator_prompt_template
    current_generator_prompt = current_generator_prompt.replace("[VALUE DEFINITION]", VALUE_DEFINITIONS)
    current_generator_prompt = current_generator_prompt.replace("[TRANSCRIPT]", transcript)

    messages = [
        {"role": "system", "content": current_generator_prompt},
    ]

    reflection = call_llm(messages, generator, seed)

    current_evaluator_prompt = evaluator_prompt_template
    current_evaluator_prompt = current_evaluator_prompt.replace("[VALUE DEFINITION]", VALUE_DEFINITIONS)
    current_evaluator_prompt = current_evaluator_prompt.replace("[TRANSCRIPT]", transcript)
    current_evaluator_prompt = current_evaluator_prompt.replace("[DIALOGUE]", reflection)

    eval_messages = [
        {"role": "system", "content": current_evaluator_prompt},
    ]

    evaluation = call_llm(eval_messages, evaluator, seed)

    return reflection, evaluation
# %%
def run_multi_turn(transcript_id, seed, generator, evaluator):

    transcript = run_transcript(transcript_id)

    current_generator_prompt = generator_prompt_template_multi
    current_generator_prompt = current_generator_prompt.replace("[VALUE DEFINITION]", VALUE_DEFINITIONS)
    current_generator_prompt = current_generator_prompt.replace("[TRANSCRIPT]", transcript)

    current_evaluator_prompt = evaluator_prompt_template_multi
    current_evaluator_prompt = current_evaluator_prompt.replace("[VALUE DEFINITION]", VALUE_DEFINITIONS)
    current_evaluator_prompt = current_evaluator_prompt.replace("[TRANSCRIPT]", transcript)

    generator_conversation = [
        {"role": "system", "content": current_generator_prompt}
    ]

    evaluator_conversation = [
        {"role": "system", "content": current_evaluator_prompt}
    ]

    log = []

    for turn in range(MAX_TURNS):
        if TERMINATE_EARLY == True:
            break
        reflection = call_llm(generator_conversation, generator, seed)

        log.append({
            "turn": turn + 1,
            "speaker": "generator",
            "content": reflection
        })

        generator_conversation.append(
            {"role": "assistant", "content": reflection}
        )

        evaluator_conversation.append(
            {"role": "user", "content": reflection}
        )

        if turn == MAX_TURNS - 1:
            break

        evaluation = call_llm(evaluator_conversation, evaluator, seed)

        log.append({
            "turn": turn + 1,
            "speaker": "evaluator",
            "content": evaluation
        })

        evaluator_conversation.append(
            {"role": "assistant", "content": evaluation}
        )

        generator_conversation.append(
            {"role": "user", "content": evaluation}
        )

    final_prompt = evaluator_prompt_template_multi_final
    final_prompt = final_prompt.replace("[VALUE DEFINITION]", VALUE_DEFINITIONS)
    final_prompt = final_prompt.replace("[TRANSCRIPT]", transcript)

    final_conversation = evaluator_conversation.copy()

    final_conversation.append({
        "role": "user",
        "content": final_prompt
    })

    final_scores = call_llm(final_conversation, evaluator, seed)

    log.append({
        "turn": "final",
        "speaker": "evaluator",
        "content": final_scores
    })

    return log

# %% 
def save_results(TRANSCRIPT_ID, seed, single_turn, multi_turn, generator, evaluator):


    MODEL_GENERATOR_SAFE = safe_id(generator)
    MODEL_EVALUATOR_SAFE = safe_id(evaluator)
    PAIR_ID = f"{MODEL_GENERATOR_SAFE}{MODEL_EVALUATOR_SAFE}"
    folder = f"experiments/transcript_{TRANSCRIPT_ID:02d}/{PAIR_ID}/seed_{seed}"
    os.makedirs(folder, exist_ok=True)

    reflection, evaluation = single_turn

    with open(f"{folder}/single_turn.txt", "w", encoding="utf-8") as f:
        f.write("=== REFLECTION ===\n")
        f.write(reflection)
        f.write("\n\n=== EVALUATION ===\n")
        f.write(evaluation)

    with open(f"{folder}/multi_turn.txt", "w", encoding="utf-8") as f:

        for entry in multi_turn:

            speaker = "LLM" if entry["speaker"] == "generator" else "Evaluator"

            f.write(f"{speaker}:\n")
            f.write(entry["content"])
            f.write("\n\n")
# %%
for gen in MODEL_GENERATORS:
    print(f"Running generator {gen}...")
    for eval in MODEL_EVALUATORS:
        print(f"Running evaluator {eval}...")
        if gen == eval:
            continue
        print("Valid Combination")
        for TRANSCRIPT_ID in TRANSCRIPTS:
            print(f"Running transcript {TRANSCRIPT_ID}...")
            for seed in SEEDS:
                print(f"Running seed {seed}...")

                single = run_single_turn(TRANSCRIPT_ID, seed, gen, eval)
                multi = run_multi_turn(TRANSCRIPT_ID, seed, gen, eval)
                save_results(TRANSCRIPT_ID, seed, single, multi, gen, eval)

print("All experiments completed.")
# %%