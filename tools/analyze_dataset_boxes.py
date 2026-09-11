"""Analyse the bounding-box size distribution of the training dataset.

Answers the deployment question: what box sizes does the model actually expect,
and are the runtime post-processing filters (YOLO_MIN_BOX_AREA,
YOLO_MAX_BOX_AREA_RATIO, YOLO_MAX_ASPECT_RATIO) discarding boxes the model was
trained to produce?

Labels are YOLO-format normalised (cx cy w h), so the width/height fractions
are resolution-independent and can be projected onto any inference resolution.
"""
import glob
import os
import sys

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'AI-MODEL',
    'Sitophilus Oryzae YOLO Detection.v2-sitophilus_oryzae_augmented_dataset_v1.0_yolov11n.yolov11',
)


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def collect(split):
    """Return (area_fractions, aspect_ratios, w_fracs, h_fracs) for one split."""
    areas, aspects, widths, heights = [], [], [], []
    pattern = os.path.join(DATASET, split, 'labels', '*.txt')
    for path in glob.glob(pattern):
        with open(path, 'r') as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 5:
                    continue
                w, h = float(parts[3]), float(parts[4])
                if w <= 0 or h <= 0:
                    continue
                areas.append(w * h)
                aspects.append(max(w, h) / min(w, h))
                widths.append(w)
                heights.append(h)
    return areas, aspects, widths, heights


def main():
    if not os.path.isdir(DATASET):
        print(f"Dataset not found: {DATASET}")
        return 1

    all_areas, all_aspects, all_w, all_h = [], [], [], []
    for split in ('train', 'valid', 'test'):
        areas, aspects, widths, heights = collect(split)
        all_areas += areas
        all_aspects += aspects
        all_w += widths
        all_h += heights
        print(f"{split:>5}: {len(areas):>6} boxes")

    if not all_areas:
        print("No boxes found.")
        return 1

    print(f"\nTotal annotated weevils: {len(all_areas)}")

    print("\n=== Box AREA as fraction of frame ===")
    for pct in (0.1, 1, 5, 25, 50, 75, 95, 99, 100):
        value = percentile(all_areas, pct)
        print(f"  p{pct:<5} {value:.6f}  ({value * 100:.4f}% of frame)")

    print("\n=== Aspect ratio (long side / short side) ===")
    for pct in (50, 75, 90, 95, 99, 100):
        print(f"  p{pct:<5} {percentile(all_aspects, pct):.2f}")

    print("\n=== Projected pixel area at each inference resolution ===")
    print("  (what the model's own trained boxes would measure at runtime)")
    for label, w_px, h_px in (
        ("640x640 (train/NCNN)", 640, 640),
        ("640x480", 640, 480),
        ("1280x720", 1280, 720),
        ("1920x1080 (current)", 1920, 1080),
    ):
        frame_px = w_px * h_px
        smallest = percentile(all_areas, 0.1) * frame_px
        p1 = percentile(all_areas, 1) * frame_px
        p50 = percentile(all_areas, 50) * frame_px
        largest = percentile(all_areas, 100) * frame_px
        print(f"\n  {label}:")
        print(f"    smallest (p0.1): {smallest:>12.0f} px^2")
        print(f"    p1             : {p1:>12.0f} px^2")
        print(f"    median         : {p50:>12.0f} px^2")
        print(f"    largest        : {largest:>12.0f} px^2")

    # Verdict against the currently configured filters.
    min_area = int(os.getenv('YOLO_MIN_BOX_AREA', '100'))
    max_ratio = float(os.getenv('YOLO_MAX_BOX_AREA_RATIO', '0.05'))
    max_aspect = float(os.getenv('YOLO_MAX_ASPECT_RATIO', '2.5'))

    print("\n=== Filter impact on the model's OWN training boxes ===")
    print(f"  Configured: min_area={min_area}px^2  "
          f"max_area_ratio={max_ratio}  max_aspect={max_aspect}")

    for label, w_px, h_px in (("640x640", 640, 640), ("1920x1080", 1920, 1080)):
        frame_px = w_px * h_px
        killed_small = sum(1 for a in all_areas if a * frame_px < min_area)
        killed_large = sum(1 for a in all_areas if a > max_ratio)
        killed_aspect = sum(1 for r in all_aspects if r > max_aspect)
        total = len(all_areas)
        print(f"\n  At {label}:")
        print(f"    rejected by min_area      : {killed_small:>6} / {total} "
              f"({100.0 * killed_small / total:.2f}%)")
        print(f"    rejected by max_area_ratio: {killed_large:>6} / {total} "
              f"({100.0 * killed_large / total:.2f}%)")
        print(f"    rejected by max_aspect    : {killed_aspect:>6} / {total} "
              f"({100.0 * killed_aspect / total:.2f}%)")

    print("\n=== Recommended filters (keep >= 99.5% of real weevils) ===")
    safe_ratio = percentile(all_areas, 100)
    safe_aspect = percentile(all_aspects, 100)
    p01_640 = percentile(all_areas, 0.1) * 640 * 640
    p01_1080 = percentile(all_areas, 0.1) * 1920 * 1080
    print(f"  YOLO_MAX_BOX_AREA_RATIO >= {safe_ratio:.4f}  (largest real weevil)")
    print(f"  YOLO_MAX_ASPECT_RATIO   >= {safe_aspect:.2f}  (most elongated real weevil)")
    print(f"  YOLO_MIN_BOX_AREA       <= {p01_640:.0f} px^2 at 640x640")
    print(f"  YOLO_MIN_BOX_AREA       <= {p01_1080:.0f} px^2 at 1920x1080")
    return 0


if __name__ == '__main__':
    sys.exit(main())
