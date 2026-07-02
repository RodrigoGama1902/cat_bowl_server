"""Configuration storage for the Cat Food Detector.

Holds the default detection parameters and persists overrides to a JSON
file on disk so the calibration done through the UI survives restarts.

The config file location can be overridden with the CONFIG_PATH env var,
which is useful for mounting a Docker volume for persistence.
"""

import copy
import json
import os

# Per-profile detection parameters (one set for day, one for night).
PROFILE_DEFAULTS = {
    "method": "texture",
    "threshold": 60,
    "dilate": 1,
    "cluster_k": 4,
    "cluster_min_texture": 0.08,
    # Cluster method: blob tone priority. "off" scores blobs by area * texture;
    # "dark" always picks the darkest qualifying blob and "bright" the brightest
    # (food is often the darkest thing in the bowl). Gates still apply.
    "cluster_tone_priority": "off",
    # Cluster/brightness geometry gates. Each can be enabled independently:
    # reject blobs touching the top edge and/or require touching the bottom.
    "cluster_reject_top_touch": False,
    "cluster_require_bottom_touch": False,
    # Cluster method: absolute maximum mean brightness (0..1) a blob may have to
    # count as food; brighter blobs are rejected. Drops dim-but-not-dark patches
    # when the food (or bowl shadow) is the darkest region. 1.0 disables it.
    "cluster_max_brightness": 1.0,
    # Brightness method: minimum spread (0..255) between the bowl's darkest and
    # brightest zones for it to count as containing food. Below this the bowl
    # is treated as empty (a clean bowl is almost uniformly bright).
    "brightness_min_contrast": 40,
    # Brightness method: drop dark blobs whose edge transition is smoother than
    # this (a lighting gradient fades gradually; real food has a crisp edge).
    # Smoothness = contrast / mean boundary gradient. 0 disables the gate.
    "brightness_max_smoothness": 0.0,
    # Close black gaps between food chunks up to this many pixels wide (a
    # morphological closing). 0 disables it.
    "fill_holes": 0,
    "minimum_coverage": 0.45,
    "full_coverage": 0.62,
}

# Built-in fallback values, used when no config file exists yet.
DEFAULTS = {
    "roi": [820, 450, 260, 180],
    "roi_shape": "rect",
    "min_artifact_area": 50,
    "profiles": {
        "day": dict(PROFILE_DEFAULTS),
        "night": dict(PROFILE_DEFAULTS),
    },
}

PROFILE_NAMES = ("day", "night")

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.json")


def _migrate(raw):
    """Upgrade an old flat config (single profile) to the day/night layout."""
    if "profiles" in raw:
        return raw
    legacy = {
        key: raw[key]
        for key in (
            "method",
            "threshold",
            "dilate",
            "cluster_k",
            "cluster_min_texture",
            "cluster_tone_priority",
            "cluster_reject_top_touch",
            "cluster_require_bottom_touch",
            "cluster_max_brightness",
            "brightness_min_contrast",
            "brightness_max_smoothness",
            "fill_holes",
            "minimum_coverage",
            "full_coverage",
        )
        if key in raw
    }
    profile = dict(PROFILE_DEFAULTS)
    profile.update(legacy)
    return {
        "roi": raw.get("roi", DEFAULTS["roi"]),
        "roi_shape": raw.get("roi_shape", DEFAULTS["roi_shape"]),
        "min_artifact_area": raw.get("min_artifact_area", DEFAULTS["min_artifact_area"]),
        "profiles": {"day": dict(profile), "night": dict(profile)},
    }


def load_config(path=None):
    """Return the current config, merging saved values over the defaults."""
    config = copy.deepcopy(DEFAULTS)
    config_path = path if path is not None else CONFIG_PATH
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as handle:
                raw = _migrate(json.load(handle))
        except (json.JSONDecodeError, OSError):
            raw = {}
        # Merge top-level keys.
        for key in ("roi", "roi_shape", "min_artifact_area"):
            if key in raw:
                config[key] = raw[key]
        # Merge each profile over the defaults so missing keys stay sane.
        for name in PROFILE_NAMES:
            if name in raw.get("profiles", {}):
                config["profiles"][name].update(raw["profiles"][name])
    return config


def get_profile(config, night_mode):
    """Return the active profile dict for the given night_mode flag."""
    return config["profiles"]["night" if night_mode else "day"]


def _write(config):
    directory = os.path.dirname(CONFIG_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return config


def save_config(updates):
    """Merge top-level updates (e.g. roi) into the saved config."""
    config = load_config()
    for key in ("roi", "roi_shape", "min_artifact_area"):
        if key in updates:
            config[key] = updates[key]
    return _write(config)


def save_profile(profile_name, updates):
    """Merge updates into a single profile (day or night) and persist."""
    config = load_config()
    if profile_name not in PROFILE_NAMES:
        raise ValueError(f"unknown profile: {profile_name}")
    config["profiles"][profile_name].update(updates)
    return _write(config)


def apply_config(updates):
    """Merge top-level and per-profile updates at once, persisting a single time.

    `updates` may contain 'roi', 'min_artifact_area', and 'profiles' (a dict of
    profile name -> partial profile dict). Missing keys keep their saved values,
    and the config file is created if it does not exist yet.
    """
    config = load_config()
    for key in ("roi", "roi_shape", "min_artifact_area"):
        if key in updates:
            config[key] = updates[key]
    for name in PROFILE_NAMES:
        if name in updates.get("profiles", {}):
            config["profiles"][name].update(updates["profiles"][name])
    return _write(config)
