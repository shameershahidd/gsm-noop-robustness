import json
import csv
import os

def create_annotation_file():
    """
    Creates a CSV file with 30 CoT responses for manual annotation.
    Run this after the full experiment is complete.
    """

    with open("results/raw_results.json", "r") as f:
        results = json.load(f)

    # We sample 30 problems for annotation
    # Take first 30 problems, CoT condition only, one model
    MODEL = "llama-3.3-70b-versatile"
    sample = results[:30]

    rows = []
    for problem in sample:
        model_result = problem["model_results"][MODEL]["cot"]
        rows.append({
            "id": problem["id"],
            "correct_answer": problem["correct_answer"],
            "noop_sentence": problem["noop_sentence"],
            "original_question": problem["original_question"],
            "noop_question": problem["noop_question"],
            "original_response": model_result["original_response"],
            "noop_response": model_result["noop_response"],
            # Annotation columns - fill these in manually
            "noop_behavior": "",      # ignores_noop / uses_noop / no_mention
            "original_correct": "",   # yes / no
            "noop_correct": "",       # yes / no
            "notes": ""               # any extra notes
        })

    output_path = "analysis/rq3_annotation.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Annotation file created: {output_path}")
    print(f"Total rows to annotate: {len(rows)}")
    print("\nAnnotation guide:")
    print("  noop_behavior:")
    print("    ignores_noop  = model explicitly identifies and discards the irrelevant sentence")
    print("    uses_noop     = model incorrectly uses the irrelevant sentence in its reasoning")
    print("    no_mention    = model doesn't mention the irrelevant sentence at all")
    print("  original_correct: yes/no")
    print("  noop_correct:     yes/no")


if __name__ == "__main__":
    create_annotation_file()