#!/usr/bin/env python3
"""Quick test: verify binary alignment + focal reward math for ESMFold.

Simulates "always predict high quality" strategy against category-imbalanced
dataset (51% high, 45% medium, 3.5% low) and checks that binary alignment
+ focal weighting correctly penalizes this reward-hacking strategy.

Expected: with binary_alignment=True, predicting "high quality" for medium/low
proteins gets 0.0 quality_alignment (no partial 0.2 credit).
"""

import json
import sys

sys.path.insert(0, ".")

from src.training.rewards import _ESMFOLD_CATEGORY_FREQ, _focal_weight, compute_esmfold_reward


def test_always_predict_high():
    """Model always says 'high quality, well-folded, plddt: 85'."""
    always_high_text = (
        "<answer>This protein exhibits high quality structure. "
        "It is well-folded and stable with high confidence. "
        "Predicted pLDDT: 85.</answer>"
    )

    # Simulate dataset distribution
    test_cases = [
        # (plddt, ptm, true_category, dataset_fraction)
        (90.0, 0.9, "high", 0.511),    # 51.1% of dataset
        (65.0, 0.7, "medium", 0.454),   # 45.4% of dataset
        (35.0, 0.4, "low", 0.035),      # 3.5% of dataset
    ]

    print("=" * 70)
    print("TEST: 'Always predict high quality' reward under different settings")
    print("=" * 70)

    for mode_name, focal_gamma, binary in [
        ("Original (no focal, no binary)", 0.0, False),
        ("Focal only (gamma=2)", 2.0, False),
        ("Binary only (no focal)", 0.0, True),
        ("Binary + Focal (gamma=2)", 2.0, True),
    ]:
        print(f"\n--- {mode_name} ---")
        weighted_total = 0.0

        for plddt, ptm, true_cat, frac in test_cases:
            metrics_json = json.dumps({"plddt": plddt, "ptm": ptm})
            reward, details = compute_esmfold_reward(
                always_high_text, metrics_json,
                detailed=True,
                focal_gamma=focal_gamma,
                binary_alignment=binary,
            )
            weighted_total += reward * frac
            print(
                f"  {true_cat:6s} (pLDDT={plddt:5.1f}, {frac*100:.1f}%): "
                f"reward={reward:.4f}  "
                f"[align={details['quality_alignment']:.2f}, "
                f"num={details['numerical_accuracy']:.2f}, "
                f"cat={details['category_match']:.2f}, "
                f"focal={details['focal_weight']:.3f}]"
            )

        print(f"  Expected dataset reward: {weighted_total:.4f}")

    print()


def test_correct_predictions():
    """Model correctly predicts each category."""
    predictions = {
        "high": (
            "<answer>This protein exhibits high quality structure. "
            "It is well-folded with high confidence. Predicted pLDDT: 88.</answer>"
        ),
        "medium": (
            "<answer>This protein has moderate structural quality. "
            "It is partially folded with medium confidence. Predicted pLDDT: 62.</answer>"
        ),
        "low": (
            "<answer>This protein is disordered and unstructured. "
            "It has low confidence fold quality. Predicted pLDDT: 30.</answer>"
        ),
    }

    test_cases = [
        (90.0, 0.9, "high", 0.511),
        (65.0, 0.7, "medium", 0.454),
        (35.0, 0.4, "low", 0.035),
    ]

    print("=" * 70)
    print("TEST: Correct predictions per category (binary + focal, gamma=2)")
    print("=" * 70)

    weighted_total = 0.0
    for plddt, ptm, true_cat, frac in test_cases:
        metrics_json = json.dumps({"plddt": plddt, "ptm": ptm})
        text = predictions[true_cat]
        reward, details = compute_esmfold_reward(
            text, metrics_json,
            detailed=True,
            focal_gamma=2.0,
            binary_alignment=True,
        )
        weighted_total += reward * frac
        print(
            f"  {true_cat:6s} (pLDDT={plddt:5.1f}, {frac*100:.1f}%): "
            f"reward={reward:.4f}  "
            f"[align={details['quality_alignment']:.2f}, "
            f"num={details['numerical_accuracy']:.2f}, "
            f"cat={details['category_match']:.2f}, "
            f"focal={details['focal_weight']:.3f}]"
        )

    print(f"  Expected dataset reward: {weighted_total:.4f}")
    print()


def test_focal_weights():
    """Print focal weights for reference."""
    print("=" * 70)
    print("Focal weights (gamma=2.0)")
    print("=" * 70)
    for cat, freq in _ESMFOLD_CATEGORY_FREQ.items():
        w = _focal_weight(cat, 2.0)
        print(f"  {cat:6s}: freq={freq:.3f}, weight={w:.3f}")
    print()


if __name__ == "__main__":
    test_focal_weights()
    test_always_predict_high()
    test_correct_predictions()

    # Quick sanity: binary_alignment should give 0.0 for well-folded claim on medium protein
    metrics_json = json.dumps({"plddt": 60.0, "ptm": 0.7})
    r_binary, d_binary = compute_esmfold_reward(
        "well-folded stable protein", metrics_json,
        detailed=True, binary_alignment=True,
    )
    r_orig, d_orig = compute_esmfold_reward(
        "well-folded stable protein", metrics_json,
        detailed=True, binary_alignment=False,
    )
    assert d_binary["quality_alignment"] == 0.0, f"Binary should give 0.0, got {d_binary['quality_alignment']}"
    assert d_orig["quality_alignment"] == 0.2, f"Original should give 0.2, got {d_orig['quality_alignment']}"
    print("PASS: Binary alignment correctly removes partial credit for medium-range proteins")
    print("PASS: Original scoring preserves partial credit")
