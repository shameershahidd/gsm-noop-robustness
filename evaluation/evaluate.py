import json
import re
import numpy as np
from scipy.stats import chi2_contingency
from collections import Counter

# Load results
with open("results/raw_results.json", "r") as f:
    results = json.load(f)

def extract_answer(text):
    """Extract final numerical answer from model response."""
    if text is None:
        return None
    
    # Look for "Final Answer: X" pattern (from CoT prompt)
    match = re.search(r'Final Answer:\s*([\d,.-]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '').strip()
    
    # Look for #### X pattern (GSM8K style)
    match = re.search(r'####\s*([\d,.-]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    
    # Look for last number in text
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    if numbers:
        return numbers[-1]
    
    return None


def extract_sc_answer(responses):
    """For self consistency, take majority answer from list of responses."""
    if not responses:
        return None
    answers = [extract_answer(r) for r in responses if r]
    answers = [a for a in answers if a is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def is_correct(predicted, correct):
    """Check if predicted answer matches correct answer."""
    if predicted is None or correct is None:
        return False
    try:
        return abs(float(predicted) - float(str(correct).replace(',', ''))) < 0.01
    except:
        return predicted.strip() == str(correct).strip()


def evaluate():
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"]
    conditions = ["standard", "cot", "sc"]
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    all_model_results = {}

    for model in models:
        print(f"\nModel: {model}")
        print("-" * 40)
        all_model_results[model] = {}

        for condition in conditions:
            orig_correct = []
            noop_correct = []

            for problem in results:
                correct_answer = problem["correct_answer"]
                model_result = problem["model_results"][model][condition]

                # Extract answers
                if condition == "sc":
                    orig_pred = extract_sc_answer(model_result["original_response"])
                    noop_pred = extract_sc_answer(model_result["noop_response"])
                else:
                    orig_pred = extract_answer(model_result["original_response"])
                    noop_pred = extract_answer(model_result["noop_response"])

                orig_correct.append(is_correct(orig_pred, correct_answer))
                noop_correct.append(is_correct(noop_pred, correct_answer))

            orig_acc = sum(orig_correct) / len(orig_correct)
            noop_acc = sum(noop_correct) / len(noop_correct)
            delta = orig_acc - noop_acc

            all_model_results[model][condition] = {
                "original_accuracy": orig_acc,
                "noop_accuracy": noop_acc,
                "delta": delta,
                "orig_correct": orig_correct,
                "noop_correct": noop_correct
            }

            print(f"  [{condition.upper()}] Original: {orig_acc:.2%} | NoOp: {noop_acc:.2%} | Delta: {delta:.2%}")

    # Save results
    with open("results/evaluation_results.json", "w") as f:
        # Convert bool lists to int for JSON serialization
        serializable = {}
        for model in all_model_results:
            serializable[model] = {}
            for condition in all_model_results[model]:
                r = all_model_results[model][condition]
                serializable[model][condition] = {
                    "original_accuracy": r["original_accuracy"],
                    "noop_accuracy": r["noop_accuracy"],
                    "delta": r["delta"]
                }
        json.dump(serializable, f, indent=2)

    print("\n\nResults saved to results/evaluation_results.json")
    return all_model_results


if __name__ == "__main__":
    evaluate()