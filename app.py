"""Flask API + configuration UI for the Cat Food Detector.

Endpoints:
    GET  /                       Configuration UI.
    GET  /health                 Liveness probe.
    POST /detect                 Run detection on an image. Use ?night_mode=true
                                 to apply the night profile (default: day).
    GET  /api/config             Current saved configuration.
    POST /api/roi                Persist the ROI ([x, y, w, h]).
    POST /api/calibrate/upload   Upload empty/medium/full images for a profile.
    GET  /api/calibrate/preview  Live detection preview for a profile's params.
    POST /api/calibrate/save     Persist a profile's threshold/method/etc.

Designed to run inside a container; mount a volume for config.json so the
calibration persists across restarts (CONFIG_PATH env var).
"""

import base64
import os
import tempfile
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from config import (
    PROFILE_NAMES,
    apply_config,
    get_profile,
    load_config,
    save_config,
    save_profile,
)
from detector import compute_mask, crop_roi, detect_image, normalize_coverage

app = Flask(__name__)

# Calibration images are stored on disk so they survive across gunicorn
# workers within the container. They are intentionally ephemeral.
CALIB_DIR = os.environ.get(
    "CALIB_DIR", os.path.join(tempfile.gettempdir(), "cat_food_calib")
)
os.makedirs(CALIB_DIR, exist_ok=True)

SLOTS = ("empty", "medium", "full")

_TRUE = {"1", "true", "yes", "on"}

# Mean saturation below this means the frame is effectively grayscale (IR night
# mode). Color daytime frames sit well above it.
NIGHT_SATURATION_MAX = 12.0

# Cache for the last successful detection result, returned by GET /status.
_status_lock = threading.Lock()
_last_status: dict | None = None


def _is_night(value, default=False):
    """Parse a night_mode flag from a string value."""
    if value is None:
        return default
    return str(value).strip().lower() in _TRUE


def _as_bool(value, default=False):
    """Parse a boolean query/JSON value, falling back to ``default`` if unset."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE


def detect_night_image(image):
    """Return True if the frame looks like night mode (near-grayscale / IR)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = float(hsv[:, :, 1].mean())
    return mean_saturation < NIGHT_SATURATION_MAX, mean_saturation


def _profile_name(value):
    """Return a valid profile name, defaulting to 'day'."""
    return value if value in PROFILE_NAMES else "day"


def _read_image_bytes():
    """Return the raw image bytes from the request, or None if absent."""
    if "image" in request.files:
        return request.files["image"].read()
    if request.data:
        return request.data
    return None


def _png_b64(image):
    """Encode an image (BGR or grayscale) as a base64 data URL."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def _calib_path(profile, slot):
    return os.path.join(CALIB_DIR, f"{profile}_{slot}.img")


# --- UI ----------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/status")
def status():
    """Return the last detection result cached by POST /detect.

    Response:
        200  {"food_present": bool, "coverage": float, "latency_ms": int,
               "last_detection": "<ISO-8601 UTC>"}
        503  {"error": "no detection yet"} when /detect has never been called.
    """
    with _status_lock:
        snapshot = _last_status
    if snapshot is None:
        return jsonify({"error": "no detection yet"}), 503
    return jsonify(snapshot)


# --- Detection ---------------------------------------------------------------
@app.post("/detect")
def detect():
    _t0 = time.monotonic()
    raw = _read_image_bytes()
    if not raw:
        return jsonify({"error": "No image provided."}), 400

    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image."}), 400

    config = load_config()
    args = request.args
    # Detection always uses the saved configuration from the UI. The only
    # request-time control is the day/night profile selection: it is
    # auto-detected from the image (grayscale = night), and an explicit
    # ?night_mode=... query param overrides that auto-detection.
    auto_night, mean_saturation = detect_night_image(image)
    override = args.get("night_mode")
    night_mode = _is_night(override) if override is not None else auto_night
    profile = get_profile(config, night_mode)
    roi = tuple(config["roi"])
    roi_shape = config.get("roi_shape", "rect")

    try:
        result = detect_image(
            image,
            roi=roi,
            threshold=profile["threshold"],
            minimum_coverage=profile["minimum_coverage"],
            min_artifact_area=config["min_artifact_area"],
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
            roi_shape=roi_shape,
            cluster_tone_priority=profile.get("cluster_tone_priority", "off"),
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    result["night_mode"] = night_mode
    result["auto_detected"] = override is None
    result["mean_saturation"] = round(mean_saturation, 1)

    latency_ms = round((time.monotonic() - _t0) * 1000)
    with _status_lock:
        global _last_status
        _last_status = {
            "food_present": result["food_present"],
            "coverage": result["coverage"],
            "latency_ms": latency_ms,
            "last_detection": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    return jsonify(result)


# --- Configuration API -------------------------------------------------------
@app.get("/api/config")
def get_config():
    return jsonify(load_config())


@app.post("/api/config")
def update_config():
    """Persist a full config (e.g. an imported preset).

    Accepts {"roi": [...], "min_artifact_area": N, "profiles": {"day": {...},
    "night": {...}}}; any omitted keys keep their saved values. Creates the
    config file if absent.
    """
    data = request.get_json(silent=True) or {}
    updates = {}

    roi = data.get("roi")
    if roi is not None:
        if len(roi) != 4:
            return jsonify({"error": "roi must be [x, y, w, h]"}), 400
        try:
            roi = [int(v) for v in roi]
        except (TypeError, ValueError):
            return jsonify({"error": "roi values must be integers"}), 400
        if roi[2] <= 0 or roi[3] <= 0:
            return jsonify({"error": "width and height must be positive"}), 400
        updates["roi"] = roi

    roi_shape = data.get("roi_shape")
    if roi_shape is not None:
        if roi_shape not in ("rect", "ellipse"):
            return jsonify({"error": "roi_shape must be 'rect' or 'ellipse'"}), 400
        updates["roi_shape"] = roi_shape

    if "min_artifact_area" in data:
        try:
            updates["min_artifact_area"] = int(data["min_artifact_area"])
        except (TypeError, ValueError):
            return jsonify({"error": "min_artifact_area must be an integer"}), 400

    profiles = data.get("profiles") or {}
    clean_profiles = {}
    for name, raw in profiles.items():
        if name not in PROFILE_NAMES or not isinstance(raw, dict):
            continue
        profile = {}
        try:
            if "method" in raw:
                if raw["method"] not in ("texture", "brightness", "cluster"):
                    return jsonify({"error": f"invalid method: {raw['method']}"}), 400
                profile["method"] = raw["method"]
            if "threshold" in raw:
                profile["threshold"] = int(raw["threshold"])
            if "minimum_coverage" in raw:
                profile["minimum_coverage"] = float(raw["minimum_coverage"])
            if "full_coverage" in raw:
                profile["full_coverage"] = float(raw["full_coverage"])
            if "dilate" in raw:
                profile["dilate"] = int(raw["dilate"])
            if "cluster_k" in raw:
                profile["cluster_k"] = int(raw["cluster_k"])
            if "cluster_min_texture" in raw:
                profile["cluster_min_texture"] = float(raw["cluster_min_texture"])
            if "cluster_tone_priority" in raw:
                if raw["cluster_tone_priority"] not in ("off", "dark", "bright"):
                    return jsonify({"error": "cluster_tone_priority must be 'off', 'dark' or 'bright'"}), 400
                profile["cluster_tone_priority"] = raw["cluster_tone_priority"]
            if "cluster_reject_top_touch" in raw:
                profile["cluster_reject_top_touch"] = bool(raw["cluster_reject_top_touch"])
            if "cluster_require_bottom_touch" in raw:
                profile["cluster_require_bottom_touch"] = bool(
                    raw["cluster_require_bottom_touch"]
                )
            if "cluster_max_brightness" in raw:
                profile["cluster_max_brightness"] = float(raw["cluster_max_brightness"])
            if "brightness_min_contrast" in raw:
                profile["brightness_min_contrast"] = int(raw["brightness_min_contrast"])
            if "fill_holes" in raw:
                profile["fill_holes"] = int(raw["fill_holes"])
            if "brightness_max_smoothness" in raw:
                profile["brightness_max_smoothness"] = float(
                    raw["brightness_max_smoothness"]
                )
        except (TypeError, ValueError):
            return jsonify({"error": f"invalid values for profile '{name}'"}), 400
        if profile:
            clean_profiles[name] = profile
    if clean_profiles:
        updates["profiles"] = clean_profiles

    if not updates:
        return jsonify({"error": "nothing to update"}), 400
    return jsonify(apply_config(updates))


@app.post("/api/roi")
def set_roi():
    data = request.get_json(silent=True) or {}
    roi = data.get("roi")
    if not roi or len(roi) != 4:
        return jsonify({"error": "roi must be [x, y, w, h]"}), 400
    try:
        roi = [int(v) for v in roi]
    except (TypeError, ValueError):
        return jsonify({"error": "roi values must be integers"}), 400
    if roi[2] <= 0 or roi[3] <= 0:
        return jsonify({"error": "width and height must be positive"}), 400
    updates = {"roi": roi}
    roi_shape = data.get("roi_shape")
    if roi_shape is not None:
        if roi_shape not in ("rect", "ellipse"):
            return jsonify({"error": "roi_shape must be 'rect' or 'ellipse'"}), 400
        updates["roi_shape"] = roi_shape
    return jsonify(save_config(updates))


@app.post("/api/calibrate/upload")
def calibrate_upload():
    profile = _profile_name(request.args.get("profile") or request.form.get("profile"))
    saved = []
    for slot in SLOTS:
        file = request.files.get(slot)
        if file and file.filename:
            file.save(_calib_path(profile, slot))
            saved.append(slot)
    if not saved:
        return jsonify({"error": "no images provided"}), 400
    return jsonify({"profile": profile, "saved": saved})


@app.get("/api/calibrate/preview")
def calibrate_preview():
    config = load_config()
    roi = tuple(config["roi"])
    profile_name = _profile_name(request.args.get("profile"))
    profile = config["profiles"][profile_name]
    threshold = request.args.get("threshold", profile["threshold"], type=int)
    minimum_coverage = request.args.get(
        "minimum_coverage", profile["minimum_coverage"], type=float
    )
    full_coverage = request.args.get(
        "full_coverage", profile["full_coverage"], type=float
    )
    method = request.args.get("method", profile["method"])
    dilate = request.args.get("dilate", profile["dilate"], type=int)
    cluster_k = request.args.get("cluster_k", profile.get("cluster_k", 4), type=int)
    cluster_min_texture = request.args.get(
        "cluster_min_texture", profile.get("cluster_min_texture", 0.08), type=float
    )
    cluster_tone_priority = request.args.get(
        "cluster_tone_priority", profile.get("cluster_tone_priority", "off")
    )
    cluster_reject_top_touch = _as_bool(
        request.args.get("cluster_reject_top_touch"),
        profile.get("cluster_reject_top_touch", False),
    )
    cluster_require_bottom_touch = _as_bool(
        request.args.get("cluster_require_bottom_touch"),
        profile.get("cluster_require_bottom_touch", False),
    )
    cluster_max_brightness = request.args.get(
        "cluster_max_brightness",
        profile.get("cluster_max_brightness", 1.0),
        type=float,
    )
    brightness_min_contrast = request.args.get(
        "brightness_min_contrast", profile.get("brightness_min_contrast", 40), type=int
    )
    fill_holes_area = request.args.get(
        "fill_holes", profile.get("fill_holes", 0), type=int
    )
    brightness_max_smoothness = request.args.get(
        "brightness_max_smoothness",
        profile.get("brightness_max_smoothness", 0.0),
        type=float,
    )
    min_artifact_area = config["min_artifact_area"]
    roi_shape = config.get("roi_shape", "rect")

    results = {}
    for slot in SLOTS:
        path = _calib_path(profile_name, slot)
        if not os.path.exists(path):
            continue
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        try:
            crop = crop_roi(image, roi)
        except ValueError as error:
            results[slot] = {"error": str(error)}
            continue
        mask = compute_mask(
            crop, threshold, min_artifact_area, method, dilate, cluster_k,
            cluster_min_texture, brightness_min_contrast, fill_holes_area,
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
        coverage = (
            round(float(np.count_nonzero(mask)) / total_pixels, 2) if total_pixels else 0.0
        )
        results[slot] = {
            "coverage": coverage,
            "normalized": round(
                normalize_coverage(coverage, minimum_coverage, full_coverage), 2
            ),
            "food_present": coverage >= minimum_coverage,
            "crop": _png_b64(crop),
            "mask": _png_b64(mask),
        }

    return jsonify(
        {
            "profile": profile_name,
            "threshold": threshold,
            "minimum_coverage": minimum_coverage,
            "full_coverage": full_coverage,
            "method": method,
            "dilate": dilate,
            "cluster_k": cluster_k,
            "cluster_min_texture": cluster_min_texture,
            "cluster_tone_priority": cluster_tone_priority,
            "cluster_reject_top_touch": cluster_reject_top_touch,
            "cluster_require_bottom_touch": cluster_require_bottom_touch,
            "cluster_max_brightness": cluster_max_brightness,
            "brightness_min_contrast": brightness_min_contrast,
            "fill_holes": fill_holes_area,
            "brightness_max_smoothness": brightness_max_smoothness,
            "results": results,
        }
    )


@app.post("/api/calibrate/save")
def calibrate_save():
    data = request.get_json(silent=True) or {}
    profile_name = _profile_name(data.get("profile"))
    updates = {}
    if "threshold" in data:
        updates["threshold"] = int(data["threshold"])
    if "minimum_coverage" in data:
        updates["minimum_coverage"] = float(data["minimum_coverage"])
    if "full_coverage" in data:
        updates["full_coverage"] = float(data["full_coverage"])
    if "method" in data and data["method"] in ("texture", "brightness", "cluster"):
        updates["method"] = data["method"]
    if "dilate" in data:
        updates["dilate"] = int(data["dilate"])
    if "cluster_k" in data:
        updates["cluster_k"] = int(data["cluster_k"])
    if "cluster_min_texture" in data:
        updates["cluster_min_texture"] = float(data["cluster_min_texture"])
    if "cluster_tone_priority" in data and data["cluster_tone_priority"] in ("off", "dark", "bright"):
        updates["cluster_tone_priority"] = data["cluster_tone_priority"]
    if "cluster_reject_top_touch" in data:
        updates["cluster_reject_top_touch"] = bool(data["cluster_reject_top_touch"])
    if "cluster_require_bottom_touch" in data:
        updates["cluster_require_bottom_touch"] = bool(
            data["cluster_require_bottom_touch"]
        )
    if "cluster_max_brightness" in data:
        updates["cluster_max_brightness"] = float(data["cluster_max_brightness"])
    if "brightness_min_contrast" in data:
        updates["brightness_min_contrast"] = int(data["brightness_min_contrast"])
    if "fill_holes" in data:
        updates["fill_holes"] = int(data["fill_holes"])
    if "brightness_max_smoothness" in data:
        updates["brightness_max_smoothness"] = float(data["brightness_max_smoothness"])
    if not updates:
        return jsonify({"error": "nothing to save"}), 400
    return jsonify(save_profile(profile_name, updates))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
