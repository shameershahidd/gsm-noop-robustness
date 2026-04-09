import json
import random
from datasets import load_dataset

random.seed(42)

# These are irrelevant sentences to inject - styled after the paper's examples
# They sound relevant but have zero effect on the answer
NOOP_SENTENCES = [
    "The temperature outside is 72 degrees Fahrenheit.",
    "John's favorite color is blue.",
    "The store was painted red last year.",
    "It was a sunny day with no clouds in the sky.",
    "The building has been standing for over 50 years.",
    "She was wearing a green hat at the time.",
    "The event took place on a Wednesday morning.",
    "The neighborhood has 12 streetlights.",
    "He had been working at the company for 3 years.",
    "The road was recently repaved last summer.",
    "There were 5 birds sitting on the fence nearby.",
    "The meeting started 10 minutes late.",
    "She had just returned from a two-week vacation.",
    "The package was delivered in a brown box.",
    "It takes 30 minutes to drive to the nearest town.",
]

# Load main dataset
print("Loading GSM-Symbolic main...")
main = load_dataset("apple/GSM-Symbolic", "main")
all_examples = list(main["test"])

# Get 100 unique problems (instance=0)
seen_ids = set()
unique_examples = []
for ex in all_examples:
    if ex["original_id"] not in seen_ids and ex["instance"] == 0:
        seen_ids.add(ex["original_id"])
        unique_examples.append(ex)

print(f"Unique problems: {len(unique_examples)}")

# Extract final answer from answer field (after ####)
def extract_final_answer(answer_text):
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return None

# Build paired dataset
paired_data = []
for i, ex in enumerate(unique_examples):
    noop_sentence = NOOP_SENTENCES[i % len(NOOP_SENTENCES)]
    noop_question = ex["question"] + " " + noop_sentence

    paired_data.append({
        "id": i,
        "original_question": ex["question"],
        "noop_question": noop_question,
        "noop_sentence": noop_sentence,
        "answer": extract_final_answer(ex["answer"]),
        "full_answer": ex["answer"],
        "original_id": ex["original_id"]
    })

# Save to file
output_path = "sampled_problems.json"
with open(output_path, "w") as f:
    json.dump(paired_data, f, indent=2)

print(f"\nSaved {len(paired_data)} pairs to {output_path}")
print("\nSample pair:")
print(f"ORIGINAL: {paired_data[0]['original_question']}")
print(f"NOOP:     {paired_data[0]['noop_question']}")
print(f"ANSWER:   {paired_data[0]['answer']}")