"""Batch-test a detector config against labelled sample directories.

Point it at two folders of snapshots — one where the bowl has food and one
where it is empty — and it reports how often the detector agrees with that
ground truth (hit rate, plus precision / recall / F1 for the "food present"
class).

The day/night profile is auto-detected per image from its mean HSV saturation
(grayscale IR frames = night), exactly like the running server. Pass
``--profile day`` or ``--profile night`` to force one profile for every image.

Usage:
    python scripts/test_accuracy.py
    python scripts/test_accuracy.py --config test_accuracy/config.json
    python scripts/test_accuracy.py --full path/to/full --empty path/to/empty
"""

import argparse
import os
import sys

import cv2

# Allow running the script directly (python scripts/test_accuracy.py) by making
# the project root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_profile, load_config  # noqa: E402
from detector import detect_image, load_image  # noqa: E402

# Mean HSV saturation below this means the frame is effectively grayscale, i.e.
# an IR night capture. Mirrors NIGHT_SATURATION_MAX in app.py.
NIGHT_SATURATION_MAX = 12.0

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_FULL = os.path.join("test_accuracy", "samples", "full")
DEFAULT_EMPTY = os.path.join("test_accuracy", "samples", "empty")


def _is_night_image(image):
    """Return True when the frame is effectively grayscale (night/IR)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) < NIGHT_SATURATION_MAX


def _list_images(directory):
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith(IMAGE_EXTS)
    ]


def _detect(image, cfg, profile):
    """Run detect_image with the parameters of a single profile."""
    return detect_image(
        image,
        roi=tuple(cfg["roi"]),
        threshold=profile["threshold"],
        minimum_coverage=profile["minimum_coverage"],
        min_artifact_area=cfg["min_artifact_area"],
        method=profile["method"],
        dilate=profile["dilate"],
        full_coverage=profile["full_coverage"],
        cluster_k=profile.get("cluster_k", 4),
        cluster_min_texture=profile.get("cluster_min_texture", 0.08),
        cluster_reject_top_touch=profile.get("cluster_reject_top_touch", False),
        cluster_require_bottom_touch=profile.get(
            "cluster_require_bottom_touch", False
        ),
        cluster_max_brightness=profile.get("cluster_max_brightness", 1.0),
        brightness_min_contrast=profile.get("brightness_min_contrast", 40),
        fill_holes_area=profile.get("fill_holes", 0),
        brightness_max_smoothness=profile.get("brightness_max_smoothness", 0.0),
        roi_shape=cfg.get("roi_shape", "rect"),
        cluster_tone_priority=profile.get("cluster_tone_priority", "off"),
    )


def _run(paths, cfg, forced_profile, label, expected_food):
    """Detect every image in ``paths`` and tally results against the label.

    Returns (correct, total, errors, night_count) where ``errors`` is a list of
    (path, result, profile_name) tuples for the misclassified images.
    """
    correct = 0
    errors = []
    night_count = 0
    for path in paths:
        try:
            image = load_image(path)
        except Exception as error:  # pragma: no cover - bad/unreadable file
            errors.append((path, {"error": str(error)}, "?"))
            continue
        if forced_profile is not None:
            profile_name = forced_profile
        else:
            profile_name = "night" if _is_night_image(image) else "day"
        if profile_name == "night":
            night_count += 1
        profile = get_profile(cfg, profile_name == "night")
        result = _detect(image, cfg, profile)
        if result["food_present"] == expected_food:
            correct += 1
        else:
            errors.append((path, result, profile_name))
    return correct, len(paths), errors, night_count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", default=DEFAULT_FULL,
        help=f"Directory of bowls WITH food (default: {DEFAULT_FULL}).",
    )
    parser.add_argument(
        "--empty", default=DEFAULT_EMPTY,
        help=f"Directory of EMPTY bowls (default: {DEFAULT_EMPTY}).",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to the config JSON to test (default: the server's CONFIG_PATH).",
    )
    parser.add_argument(
        "--profile", choices=("day", "night"), default=None,
        help="Force a profile for every image (default: auto-detect per image).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(path=args.config)

    full_paths = _list_images(args.full)
    empty_paths = _list_images(args.empty)
    if not full_paths and not empty_paths:
        parser.error(
            f"No images found in '{args.full}' or '{args.empty}'."
        )

    # food_present is True for the "full" set and False for the "empty" set.
    full_correct, full_total, full_errors, full_night = _run(
        full_paths, cfg, args.profile, "full", True
    )
    empty_correct, empty_total, empty_errors, empty_night = _run(
        empty_paths, cfg, args.profile, "empty", False
    )

    # Confusion matrix for the "food present" class.
    tp = full_correct              # food images correctly flagged as food
    fn = full_total - full_correct  # food images missed (called empty)
    tn = empty_correct             # empty images correctly flagged as empty
    fp = empty_total - empty_correct  # empty images wrongly flagged as food

    total = full_total + empty_total
    correct = tp + tn
    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    print(f"Config:   {args.config or os.environ.get('CONFIG_PATH', 'config.json')}")
    if args.profile:
        print(f"Profile:  forced '{args.profile}'")
    else:
        print(
            f"Profile:  auto ({full_night + empty_night}/{total} detected as night)"
        )
    print()
    print(f"Full  dir ({args.full}): {full_correct}/{full_total} correct")
    print(f"Empty dir ({args.empty}): {empty_correct}/{empty_total} correct")
    print()
    print(f"Accuracy:  {accuracy:.1%} ({correct}/{total})")
    print(f"Precision: {precision:.1%}")
    print(f"Recall:    {recall:.1%}")
    print(f"F1 score:  {f1:.3f}")
    print(f"Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")

    _print_errors("Empty check", args.empty, empty_total, empty_errors)
    _print_errors("Full check", args.full, full_total, full_errors)


def _print_errors(title, directory, total, errors):
    """Print the misclassified images for one labelled set."""
    print()
    print(f"=== {title} ({directory}) — {total - len(errors)}/{total} correct ===")
    if not errors:
        print("  All correct.")
        return
    print(f"Misclassified ({len(errors)}):")
    for path, res, mode in errors:
        if "error" in res:
            print(f"  [{mode:5}] ERROR: {res['error']}")
        else:
            print(
                f"  [{mode:5}] food_present={res['food_present']} "
                f"raw={res['raw_coverage']:.3f} coverage={res['coverage']:.3f}"
            )
        # Absolute path on its own line so it is Ctrl+Click-able in VS Code.
        print(f"    {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
