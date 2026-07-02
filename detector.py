"""Cat Food Detector.

Lightweight, local, no-ML computer vision script that estimates how much
food is present in a cat food bowl from a single image snapshot.

Usage:
    python detector.py latest.jpg

Output (stdout, JSON):
    {"food_present": true, "coverage": 0.63}
"""

import argparse
import json
import sys

import cv2
import numpy as np

from config import DEFAULTS

# --- Configuration -----------------------------------------------------------
# Module-level defaults come from config.py's "day" profile; the UI persists
# per-profile overrides (day/night) to config.json.
_DAY = DEFAULTS["profiles"]["day"]

# Region of Interest as (x, y, width, height) in pixels.
ROI = tuple(DEFAULTS["roi"])

# Grayscale threshold: for the brightness method, pixels darker than this are
# treated as food; for the texture method, this is the Canny edge sensitivity.
THRESHOLD = _DAY["threshold"]

# Coverage above which the bowl is considered to contain food.
MINIMUM_COVERAGE = _DAY["minimum_coverage"]

# Raw coverage that represents a full bowl. Raw coverage is remapped so that
# MINIMUM_COVERAGE -> 0.0 and FULL_COVERAGE -> 1.0 (clamped in between).
FULL_COVERAGE = _DAY["full_coverage"]

# Minimum artifact area (in pixels) to keep. Smaller blobs are removed as noise.
# Set to 0 to disable artifact removal.
MIN_ARTIFACT_AREA = DEFAULTS["min_artifact_area"]

# Detection method: "texture" (edge density), "brightness", or "cluster"
# (largest homogeneous color/texture blob).
METHOD = _DAY["method"]

# Dilation iterations used to fill textured (food) regions in the texture method.
DILATE = _DAY["dilate"]

# Number of color groups used by the "cluster" method. Pixels are quantized into
# this many groups (k-means in LAB space); food is the largest contiguous group.
CLUSTER_K = DEFAULTS.get("cluster_k", 4)

# Minimum granularity (edge density, 0..1) for a blob to count as food in the
# "cluster" method. A smooth bowl bottom has almost no edges and is rejected;
# the granular kibble pile passes. Set to 0 to accept any blob.
CLUSTER_MIN_TEXTURE = _DAY.get("cluster_min_texture", 0.08)

# Cluster method: blob selection. "off" keeps the neutral area * texture score;
# "dark" always picks the darkest qualifying blob and "bright" the brightest,
# ignoring area/texture for the win decision (gates still apply).
CLUSTER_TONE_PRIORITY = _DAY.get("cluster_tone_priority", "off")

# Fixed seed for the cluster method's k-means so detection is deterministic.
_CLUSTER_RNG_SEED = 42

# Cluster/brightness geometry gates, configurable independently.
# Reject blobs touching the top edge and/or require touching the bottom edge.
CLUSTER_REJECT_TOP_TOUCH = _DAY.get("cluster_reject_top_touch", False)
CLUSTER_REQUIRE_BOTTOM_TOUCH = _DAY.get("cluster_require_bottom_touch", False)

# Cluster method: absolute maximum mean brightness (0..1) a blob may have to
# qualify as food. Blobs brighter than this are rejected outright, which drops
# dim-but-not-dark patches when the food (or bowl shadow) is the darkest region.
# 1.0 disables the gate (accept any brightness).
CLUSTER_MAX_BRIGHTNESS = _DAY.get("cluster_max_brightness", 1.0)

# Brightness method: minimum spread (0..255) between the bowl's darkest and
# brightest zones for it to count as food. Below this the bowl is treated as
# empty (a clean bowl is almost uniformly bright).
BRIGHTNESS_MIN_CONTRAST = _DAY.get("brightness_min_contrast", 40)

# Brightness method: maximum "smoothness" allowed for a food blob's edge. A real
# food/bowl boundary is abrupt (sharp), while a lighting gradient is gradual.
# Smoothness = contrast / edge sharpness; blobs smoother than this are dropped.
# 0 disables the smoothness gate.
BRIGHTNESS_MAX_SMOOTHNESS = _DAY.get("brightness_max_smoothness", 0.0)

# Close black gaps between food chunks up to this many pixels wide (a
# morphological closing). 0 disables it.
FILL_HOLES = _DAY.get("fill_holes", 0)
# -----------------------------------------------------------------------------


def load_image(path):
    """Load an image from disk, raising a clear error if it cannot be read."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def crop_roi(image, roi):
    """Crop the configured ROI, clamped to the image bounds."""
    x, y, w, h = roi
    height, width = image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"ROI {roi} is outside the image bounds {width}x{height}")
    return image[y0:y1, x0:x1]


def compute_mask(
    crop,
    threshold=THRESHOLD,
    min_artifact_area=MIN_ARTIFACT_AREA,
    method=METHOD,
    dilate=DILATE,
    cluster_k=CLUSTER_K,
    cluster_min_texture=CLUSTER_MIN_TEXTURE,
    brightness_min_contrast=BRIGHTNESS_MIN_CONTRAST,
    fill_holes_area=FILL_HOLES,
    brightness_max_smoothness=BRIGHTNESS_MAX_SMOOTHNESS,
    cluster_reject_top_touch=CLUSTER_REJECT_TOP_TOUCH,
    cluster_require_bottom_touch=CLUSTER_REQUIRE_BOTTOM_TOUCH,
    cluster_max_brightness=CLUSTER_MAX_BRIGHTNESS,
    cluster_tone_priority=CLUSTER_TONE_PRIORITY,
):
    """Return the binary food mask for the cropped region."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if method == "brightness":
        # Dark pixels are food, but only if the bowl shows enough contrast and
        # the blob's edge is abrupt (not a smooth lighting gradient).
        mask = _brightness_mask(
            gray, threshold, brightness_min_contrast, brightness_max_smoothness
        )
        if cluster_reject_top_touch or cluster_require_bottom_touch:
            # Same geometric gates as cluster mode, with independent toggles.
            mask = _filter_by_top_bottom(
                mask, cluster_reject_top_touch, cluster_require_bottom_touch
            )
    elif method == "cluster":
        # Food forms one large homogeneous color/texture blob; isolate it.
        mask = _cluster_mask(
            crop, cluster_k, dilate, cluster_min_texture,
            cluster_reject_top_touch,
            cluster_require_bottom_touch,
            cluster_max_brightness,
            cluster_tone_priority,
            min_artifact_area,
        )
    else:
        # Texture: food (kibble) creates dense edges; an empty bowl is smooth.
        mask = _texture_mask(gray, threshold, dilate)

    if min_artifact_area > 0:
        mask = remove_small_artifacts(mask, min_artifact_area)
    if fill_holes_area > 0:
        mask = fill_holes(mask, fill_holes_area)
    return mask


def _brightness_mask(gray, threshold, min_contrast, max_smoothness=0.0):
    """Mask the dark food region using a contrast-gated, adaptive threshold.

    The bowl's contrast is the spread between its darkest and brightest zones,
    measured with robust 5th/95th percentiles so a few stray pixels do not skew
    it. When that spread is below ``min_contrast`` the bowl is treated as empty
    (a clean bowl is almost uniformly bright, so there is little to separate).

    Otherwise the cut point is placed *inside* the measured brightness band, so
    it self-adjusts to each lighting condition instead of using a fixed grey
    level: ``threshold`` (0..255) slides the cut between the dark floor and the
    bright ceiling of that band. Pixels darker than the cut are food, because
    the food is always the darkest color in the bowl.

    Finally, when ``max_smoothness`` > 0 each candidate dark blob is checked for
    edge sharpness: a real food/bowl boundary is abrupt, while a shadow or
    lighting gradient fades gradually. Blobs whose transition is smoother than
    ``max_smoothness`` (i.e. contrast divided by their mean boundary gradient)
    are discarded as lighting artifacts.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dark = float(np.percentile(blurred, 5))
    bright = float(np.percentile(blurred, 95))
    contrast = bright - dark
    if contrast < min_contrast:
        # Not enough contrast: nothing stands out as food.
        return np.zeros(gray.shape[:2], dtype=np.uint8)
    ratio = min(1.0, max(0.0, threshold / 255.0))
    cut = dark + contrast * ratio
    _, mask = cv2.threshold(gray, cut, 255, cv2.THRESH_BINARY_INV)
    if max_smoothness > 0:
        mask = _drop_smooth_blobs(mask, blurred, contrast, max_smoothness)
    return mask


def _drop_smooth_blobs(mask, blurred, contrast, max_smoothness):
    """Keep only blobs whose edge is sharp enough to be a real food boundary.

    For each connected component, the mean image-gradient magnitude along its
    boundary measures how abrupt its edge is. ``smoothness = contrast / edge``
    is large for a gradual fade (lighting) and small for a crisp food edge.
    Blobs with ``smoothness > max_smoothness`` are removed.
    """
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    kept = np.zeros_like(mask)
    for label in range(1, num_labels):
        blob = np.where(labels == label, 255, 0).astype(np.uint8)
        boundary = cv2.morphologyEx(blob, cv2.MORPH_GRADIENT, kernel)
        edge_pixels = magnitude[boundary > 0]
        if edge_pixels.size == 0:
            # No measurable boundary (e.g. blob spans the whole crop): keep it.
            kept[labels == label] = 255
            continue
        mean_edge = float(edge_pixels.mean())
        smoothness = contrast / (mean_edge + 1e-6)
        if smoothness <= max_smoothness:
            kept[labels == label] = 255
    return kept


def _texture_mask(gray, edge_threshold, dilate):
    """Build a mask of textured regions using Canny edges plus morphology."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, edge_threshold, edge_threshold * 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if dilate > 0:
        edges = cv2.dilate(edges, kernel, iterations=dilate)
    # Close gaps so scattered edges merge into solid food blobs.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges


def _filter_by_top_bottom(mask, reject_top_touch=False, require_bottom_touch=False):
    """Filter blobs by top/bottom edge contact with independent gates.

    ``reject_top_touch`` drops blobs whose bounding box touches the first row.
    ``require_bottom_touch`` drops blobs whose bounding box does not reach the
    last row. Both can be enabled together.
    """
    if not reject_top_touch and not require_bottom_touch:
        return mask
    height = mask.shape[0]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask)
    for label in range(1, num):
        top = stats[label, cv2.CC_STAT_TOP]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT]
        if require_bottom_touch and bottom < height:
            continue
        if reject_top_touch and top <= 0:
            continue
        kept[labels == label] = 255
    return kept


def _cluster_mask(
    crop,
    cluster_k,
    dilate,
    min_texture=CLUSTER_MIN_TEXTURE,
    reject_top_touch=CLUSTER_REJECT_TOP_TOUCH,
    require_bottom_touch=CLUSTER_REQUIRE_BOTTOM_TOUCH,
    max_brightness=CLUSTER_MAX_BRIGHTNESS,
    tone_priority=CLUSTER_TONE_PRIORITY,
    min_blob_area=1,
):
    """Mask the food blob via color clustering gated by minimum granularity.

    Pixels are quantized into ``cluster_k`` color groups with k-means in LAB
    space (perceptually closer to how similar colors look). Each group is split
    into connected components. A blob only qualifies as food if its local edge
    density (granularity) reaches ``min_texture`` — this rejects the smooth bowl
    bottom/rim, which has almost no edges, while the granular kibble passes.

    ``tone_priority`` controls how the winning blob is chosen among the
    qualifying ones. With "off" the score is ``area * texture`` (neutral). With
    "dark" the darkest qualifying blob always wins and with "bright" the
    brightest, regardless of area or texture (the gates still apply). This is
    useful when the food is reliably the darkest (or brightest) thing in the
    bowl.

    Top/bottom geometry gates can be enabled independently. With
    ``reject_top_touch`` blobs touching the top edge are discarded. With
    ``require_bottom_touch`` blobs not reaching the bottom edge are discarded.

    ``max_brightness`` (0..1) is an absolute cap on a blob's mean brightness: any
    blob brighter than it is rejected outright, which drops dim-but-not-dark
    patches when the food (or bowl shadow) is the darkest region. 1.0 disables
    the cap.
    """
    k = max(2, int(cluster_k))
    # Blur first so kibble texture/noise does not fragment the color groups.
    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    samples = lab.reshape((-1, 3)).astype(np.float32)

    # k cannot exceed the number of available samples.
    k = min(k, samples.shape[0])
    if k < 2:
        return np.zeros(crop.shape[:2], dtype=np.uint8)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # k-means seeds its centers randomly, so the same crop can otherwise yield a
    # different partition (and a different "darkest" blob) on each run. Seed the
    # RNG from a fixed value before clustering so detection is reproducible.
    cv2.setRNGSeed(_CLUSTER_RNG_SEED)
    _, labels, _ = cv2.kmeans(
        samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(crop.shape[:2])

    # Per-pixel texture map: edges from the (unblurred) grayscale image mark the
    # busy kibble surface. A smooth bowl bottom has almost none.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    best_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    tone_priority = (tone_priority or "off").lower()
    hard_tone = tone_priority in ("dark", "bright")
    # For "off" we maximize area * texture. For "dark"/"bright" we track the
    # extreme mean brightness instead, so the win is decided purely by tone.
    best_score = 0.0
    best_tone = 2.0 if tone_priority == "dark" else -1.0
    # A candidate blob must be at least as large as the artifact-removal
    # threshold so the winner cannot be wiped by remove_small_artifacts after
    # _cluster_mask returns, leaving an all-black mask.
    min_area = max(min_blob_area, max(1, int(0.002 * crop.shape[0] * crop.shape[1])))
    crop_h = crop.shape[0]
    for label in range(k):
        group = np.where(labels == label, 255, 0).astype(np.uint8)
        group = cv2.morphologyEx(group, cv2.MORPH_CLOSE, kernel)
        num, comps, stats, _ = cv2.connectedComponentsWithStats(group, connectivity=8)
        for comp in range(1, num):
            area = stats[comp, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            top = stats[comp, cv2.CC_STAT_TOP]
            bottom = top + stats[comp, cv2.CC_STAT_HEIGHT]
            if require_bottom_touch and bottom < crop_h:
                continue
            if reject_top_touch and top <= 0:
                continue
            blob = comps == comp
            # Texture density inside the blob (0..1): fraction of edge pixels.
            texture = float(np.count_nonzero(edges[blob])) / area
            if texture < min_texture:
                continue
            mean_brightness = float(gray[blob].mean()) / 255.0
            # Absolute brightness cap: drop blobs brighter than the allowed
            # maximum (e.g. dim patches that are not as dark as the bowl
            # shadow). 1.0 disables the gate.
            if max_brightness < 1.0 and mean_brightness > max_brightness:
                continue
            if hard_tone:
                # The darkest (or brightest) qualifying blob wins outright.
                if tone_priority == "dark":
                    if mean_brightness < best_tone:
                        best_tone = mean_brightness
                        best_mask = np.where(blob, 255, 0).astype(np.uint8)
                else:
                    if mean_brightness > best_tone:
                        best_tone = mean_brightness
                        best_mask = np.where(blob, 255, 0).astype(np.uint8)
            else:
                # Neutral score: prefer the largest, most textured blob.
                score = area * texture
                if score > best_score:
                    best_score = score
                    best_mask = np.where(blob, 255, 0).astype(np.uint8)

    if dilate > 0:
        best_mask = cv2.dilate(best_mask, kernel, iterations=dilate)
    return best_mask


def compute_coverage(
    crop,
    threshold=THRESHOLD,
    min_artifact_area=MIN_ARTIFACT_AREA,
    method=METHOD,
    dilate=DILATE,
    cluster_k=CLUSTER_K,
    cluster_min_texture=CLUSTER_MIN_TEXTURE,
    brightness_min_contrast=BRIGHTNESS_MIN_CONTRAST,
    fill_holes_area=FILL_HOLES,
    brightness_max_smoothness=BRIGHTNESS_MAX_SMOOTHNESS,
    cluster_reject_top_touch=CLUSTER_REJECT_TOP_TOUCH,
    cluster_require_bottom_touch=CLUSTER_REQUIRE_BOTTOM_TOUCH,
    cluster_max_brightness=CLUSTER_MAX_BRIGHTNESS,
    roi_shape="rect",
    cluster_tone_priority=CLUSTER_TONE_PRIORITY,
):
    """Return the fraction of food pixels in the cropped region."""
    mask = compute_mask(
        crop,
        threshold,
        min_artifact_area,
        method,
        dilate,
        cluster_k,
        cluster_min_texture,
        brightness_min_contrast,
        fill_holes_area,
        brightness_max_smoothness,
        cluster_reject_top_touch,
        cluster_require_bottom_touch,
        cluster_max_brightness,
        cluster_tone_priority,
    )
    if roi_shape == "ellipse":
        h, w = mask.shape[:2]
        shape_mask = np.zeros_like(mask)
        cv2.ellipse(shape_mask, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)), 0, 0, 360, 255, -1)
        mask = cv2.bitwise_and(mask, shape_mask)
        total_pixels = int(np.count_nonzero(shape_mask))
    else:
        total_pixels = mask.size
    food_pixels = int(np.count_nonzero(mask))
    if total_pixels == 0:
        return 0.0
    return food_pixels / total_pixels


def remove_small_artifacts(mask, min_area):
    """Remove connected components smaller than min_area from a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    # Label 0 is the background; skip it.
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def fill_holes(mask, gap):
    """Close black gaps between food chunks up to ``gap`` pixels wide.

    Uses a morphological closing (dilate then erode) with a circular kernel of
    diameter ``gap``. This bridges black gaps *between* nearby white chunks and
    fills small enclosed holes alike — unlike an enclosed-hole fill, it does not
    care whether the gap connects to the outer background through thin channels.
    Larger gaps survive because the kernel cannot span them.
    """
    if gap <= 0:
        return mask
    # Kernel diameter ~= the widest gap to bridge; keep it odd and >= 3.
    size = max(3, int(gap))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def normalize_coverage(raw, minimum_coverage, full_coverage):
    """Remap raw coverage to 0-1 using the empty/full calibration bounds."""
    if full_coverage <= minimum_coverage:
        return 1.0 if raw >= full_coverage else 0.0
    normalized = (raw - minimum_coverage) / (full_coverage - minimum_coverage)
    return min(1.0, max(0.0, normalized))


def detect_image(
    image,
    roi=ROI,
    threshold=THRESHOLD,
    minimum_coverage=MINIMUM_COVERAGE,
    min_artifact_area=MIN_ARTIFACT_AREA,
    method=METHOD,
    dilate=DILATE,
    full_coverage=FULL_COVERAGE,
    cluster_k=CLUSTER_K,
    cluster_min_texture=CLUSTER_MIN_TEXTURE,
    brightness_min_contrast=BRIGHTNESS_MIN_CONTRAST,
    fill_holes_area=FILL_HOLES,
    brightness_max_smoothness=BRIGHTNESS_MAX_SMOOTHNESS,
    cluster_reject_top_touch=CLUSTER_REJECT_TOP_TOUCH,
    cluster_require_bottom_touch=CLUSTER_REQUIRE_BOTTOM_TOUCH,
    cluster_max_brightness=CLUSTER_MAX_BRIGHTNESS,
    roi_shape="rect",
    cluster_tone_priority=CLUSTER_TONE_PRIORITY,
):
    """Run the detection pipeline on a decoded image (BGR numpy array)."""
    crop = crop_roi(image, roi)
    raw_coverage = compute_coverage(
        crop, threshold, min_artifact_area, method, dilate, cluster_k, cluster_min_texture,
        brightness_min_contrast, fill_holes_area, brightness_max_smoothness,
        cluster_reject_top_touch,
        cluster_require_bottom_touch,
        cluster_max_brightness,
        roi_shape=roi_shape, cluster_tone_priority=cluster_tone_priority,
    )
    coverage = normalize_coverage(raw_coverage, minimum_coverage, full_coverage)
    return {
        "food_present": raw_coverage >= minimum_coverage,
        "coverage": round(coverage, 2),
        "raw_coverage": round(raw_coverage, 2),
    }


def detect(
    path,
    roi=ROI,
    threshold=THRESHOLD,
    minimum_coverage=MINIMUM_COVERAGE,
    min_artifact_area=MIN_ARTIFACT_AREA,
    method=METHOD,
    dilate=DILATE,
    full_coverage=FULL_COVERAGE,
    cluster_k=CLUSTER_K,
    cluster_min_texture=CLUSTER_MIN_TEXTURE,
    brightness_min_contrast=BRIGHTNESS_MIN_CONTRAST,
    fill_holes_area=FILL_HOLES,
    brightness_max_smoothness=BRIGHTNESS_MAX_SMOOTHNESS,
    cluster_reject_top_touch=CLUSTER_REJECT_TOP_TOUCH,
    cluster_require_bottom_touch=CLUSTER_REQUIRE_BOTTOM_TOUCH,
    cluster_max_brightness=CLUSTER_MAX_BRIGHTNESS,
    roi_shape="rect",
    cluster_tone_priority=CLUSTER_TONE_PRIORITY,
):
    """Run the full detection pipeline and return the result dictionary."""
    image = load_image(path)
    return detect_image(
        image,
        roi,
        threshold,
        minimum_coverage,
        min_artifact_area,
        method,
        dilate,
        full_coverage,
        cluster_k,
        cluster_min_texture,
        brightness_min_contrast,
        fill_holes_area,
        brightness_max_smoothness,
        cluster_reject_top_touch,
        cluster_require_bottom_touch,
        cluster_max_brightness,
        roi_shape=roi_shape,
        cluster_tone_priority=cluster_tone_priority,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect cat food in a bowl snapshot.")
    parser.add_argument("image", help="Path to the snapshot image.")
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=ROI,
        help="Region of Interest as X Y WIDTH HEIGHT in pixels.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=THRESHOLD,
        help="Brightness threshold or, for texture, the Canny edge sensitivity.",
    )
    parser.add_argument(
        "--method",
        choices=("texture", "brightness", "cluster"),
        default=METHOD,
        help="Detection method: texture, brightness, or cluster.",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=DILATE,
        help="Dilation iterations for the texture/cluster methods.",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=CLUSTER_K,
        help="Number of color groups for the cluster method.",
    )
    parser.add_argument(
        "--cluster-min-texture",
        type=float,
        default=CLUSTER_MIN_TEXTURE,
        help="Minimum granularity (edge density 0..1) for a cluster blob.",
    )
    parser.add_argument(
        "--brightness-min-contrast",
        type=int,
        default=BRIGHTNESS_MIN_CONTRAST,
        help="Brightness: minimum dark/bright spread (0..255) to count as food.",
    )
    parser.add_argument(
        "--brightness-max-smoothness",
        type=float,
        default=BRIGHTNESS_MAX_SMOOTHNESS,
        help="Brightness: drop dark blobs whose edge is smoother than this; 0 off.",
    )
    parser.add_argument(
        "--fill-holes",
        type=int,
        default=FILL_HOLES,
        help="Close black gaps between food chunks up to this many px; 0 disables.",
    )
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=MINIMUM_COVERAGE,
        help="Raw coverage mapped to 0.0 (empty bowl floor).",
    )
    parser.add_argument(
        "--full-coverage",
        type=float,
        default=FULL_COVERAGE,
        help="Raw coverage mapped to 1.0 (full bowl ceiling).",
    )
    parser.add_argument(
        "--min-artifact-area",
        type=int,
        default=MIN_ARTIFACT_AREA,
        help="Minimum blob area (pixels) to keep; 0 disables artifact removal.",
    )
    args = parser.parse_args(argv)

    try:
        result = detect(
            args.image,
            roi=tuple(args.roi),
            threshold=args.threshold,
            minimum_coverage=args.minimum_coverage,
            min_artifact_area=args.min_artifact_area,
            method=args.method,
            dilate=args.dilate,
            full_coverage=args.full_coverage,
            cluster_k=args.cluster_k,
            cluster_min_texture=args.cluster_min_texture,
            brightness_min_contrast=args.brightness_min_contrast,
            fill_holes_area=args.fill_holes,
            brightness_max_smoothness=args.brightness_max_smoothness,
        )
    except (FileNotFoundError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
