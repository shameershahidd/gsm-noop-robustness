import json
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import bootstrap as scipy_bootstrap

# Load evaluation results
with open("results/raw_results.json", "r") as f:
    raw_results = json.load(f)

# Re-run evaluation to get per-problem correct/incorrect lists
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluate import evaluate

all_model_results = evaluate()

def mcnemar_test(condition1_correct, condition2_correct):
    """
    McNemar's test comparing two conditions.
    condition1_correct, condition2_correct: lists of booleans
    """
    # Build contingency table
    both_correct = sum(a and b for a, b in zip(condition1_correct, condition2_correct))
    only_1_correct = sum(a and not b for a, b in zip(condition1_correct, condition2_correct))
    only_2_correct = sum(not a and b for a, b in zip(condition1_correct, condition2_correct))
    neither_correct = sum(not a and not b for a, b in zip(condition1_correct, condition2_correct))

    table = [[both_correct, only_1_correct],
             [only_2_correct, neither_correct]]

    # Use chi2 with correction
    if only_1_correct + only_2_correct == 0:
        return 1.0  # No difference
    
    statistic = (abs(only_1_correct - only_2_correct) - 1) ** 2 / (only_1_correct + only_2_correct)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, df=1)
    return p_value


def bootstrap_confidence_interval(correct_list, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for accuracy."""
    correct_array = np.array(correct_list, dtype=float)
    means = []
    n = len(correct_array)
    for _ in range(n_bootstrap):
        sample = np.random.choice(correct_array, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def run_statistics():
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"]
    conditions = ["standard", "cot", "sc"]

    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    stats_results = {}

    for model in models:
        print(f"\nModel: {model}")
        print("-" * 40)
        stats_results[model] = {}

        for condition in conditions:
            r = all_model_results[model][condition]
            orig_correct = r["orig_correct"]
            noop_correct = r["noop_correct"]

            # Bootstrap CIs
            orig_lower, orig_upper = bootstrap_confidence_interval(orig_correct)
            noop_lower, noop_upper = bootstrap_confidence_interval(noop_correct)

            # McNemar's test: original vs noop for this condition
            p_value = mcnemar_test(orig_correct, noop_correct)

            print(f"  [{condition.upper()}]")
            print(f"    Original Acc: {r['original_accuracy']:.2%} (95% CI: {orig_lower:.2%} - {orig_upper:.2%})")
            print(f"    NoOp Acc:     {r['noop_accuracy']:.2%} (95% CI: {noop_lower:.2%} - {noop_upper:.2%})")
            print(f"    Delta:        {r['delta']:.2%}")
            print(f"    McNemar p-value (orig vs noop): {p_value:.4f}")

            stats_results[model][condition] = {
                "original_accuracy": r["original_accuracy"],
                "noop_accuracy": r["noop_accuracy"],
                "delta": r["delta"],
                "orig_ci": [orig_lower, orig_upper],
                "noop_ci": [noop_lower, noop_upper],
                "mcnemar_p_value": p_value
            }

        # McNemar: standard vs cot on noop questions (RQ2)
        p_std_vs_cot = mcnemar_test(
            all_model_results[model]["standard"]["noop_correct"],
            all_model_results[model]["cot"]["noop_correct"]
        )
        p_std_vs_sc = mcnemar_test(
            all_model_results[model]["standard"]["noop_correct"],
            all_model_results[model]["sc"]["noop_correct"]
        )

        print(f"\n  RQ2 - McNemar Standard vs CoT on NoOp: p={p_std_vs_cot:.4f}")
        print(f"  RQ2 - McNemar Standard vs SC on NoOp:  p={p_std_vs_sc:.4f}")

    # Save
    with open("results/statistics_results.json", "w") as f:
        json.dump(stats_results, f, indent=2)
    print("\n\nStatistics saved to results/statistics_results.json")


if __name__ == "__main__":
    run_statistics()