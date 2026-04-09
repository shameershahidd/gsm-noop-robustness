import json
import os
import time
import sys
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.run_inference import call_model, run_self_consistency
from prompts.templates import standard_prompt, cot_prompt, sc_prompt

load_dotenv()

# Load problems
with open("data/sampled_problems.json", "r") as f:
    problems = json.load(f)

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b"
]

def run_experiment(test_mode=False):
    """
    Run the full experiment across all models, conditions, and problems.
    test_mode=True runs only first 3 problems for quick testing.
    """
    
    problems_to_run = problems[:3] if test_mode else problems
    results = []

    print(f"Running experiment on {len(problems_to_run)} problems...")
    print(f"Models: {MODELS}")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}\n")

    for i, problem in enumerate(problems_to_run):
        print(f"Problem {i+1}/{len(problems_to_run)} (id={problem['id']})")
        result = {
            "id": problem["id"],
            "original_question": problem["original_question"],
            "noop_question": problem["noop_question"],
            "correct_answer": problem["answer"],
            "noop_sentence": problem["noop_sentence"],
            "model_results": {}
        }

        for model in MODELS:
            print(f"  Model: {model}")
            result["model_results"][model] = {}

            for condition in ["standard", "cot", "sc"]:
                # Get the right prompt for original and noop questions
                if condition == "standard":
                    orig_prompt = standard_prompt(problem["original_question"])
                    noop_prompt = standard_prompt(problem["noop_question"])
                elif condition == "cot":
                    orig_prompt = cot_prompt(problem["original_question"])
                    noop_prompt = cot_prompt(problem["noop_question"])
                else:  # sc
                    orig_prompt = sc_prompt(problem["original_question"])
                    noop_prompt = sc_prompt(problem["noop_question"])

                # Run original question
                if condition == "sc":
                    orig_responses = run_self_consistency(model, orig_prompt, n=5)
                    noop_responses = run_self_consistency(model, noop_prompt, n=5)
                    orig_response = orig_responses
                    noop_response = noop_responses
                else:
                    orig_response = call_model(model, orig_prompt)
                    noop_response = call_model(model, noop_prompt)
                    time.sleep(0.5)

                result["model_results"][model][condition] = {
                    "original_response": orig_response,
                    "noop_response": noop_response
                }

                print(f"    {condition} ✓")

        results.append(result)

        # Save after every problem in case of crash
        output_path = "results/raw_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone! Results saved to {output_path}")
    return results


if __name__ == "__main__":
    # Run in test mode first (3 problems only)
    run_experiment(test_mode=True)