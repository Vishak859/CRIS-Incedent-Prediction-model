"""
Rail Failure Analysis System — Streamlit App
=============================================
Module 1 : Random Forest Remaining Life Predictor (all-India)
           Artifacts: random_forest_model.pkl  preprocessor.pkl  section_freq.pkl

Module 2 : BNN-Cox Survival & Risk Analyser (NCR zone)
           Artifacts: bnn_model_NCR.h5  cph_NCR.pkl  scaler_NCR.pkl
"""

import json
import pickle
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib.gridspec import GridSpec
from requests.auth import HTTPBasicAuth

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rail Failure Analysis System",
    page_icon="🛤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
  --dark:#ffffff; --surface:#f6f8fb; --card:#ffffff; --border:#d7dee8;
  --amber:#0f766e; --green:#15803d; --red:#dc2626; --blue:#1d4ed8;
  --purple:#6d28d9; --muted:#475569; --text:#0f172a;
}
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--dark);
  color: var(--text);
  font-size: 16px;
  font-weight: 500;
}
.stApp { background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); }

/* Header */
.app-header {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 30px 36px;
  margin-bottom: 28px;
  box-shadow: 0 8px 24px rgba(15,23,42,.07);
  position: relative;
  overflow: hidden;
}
.app-header::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, #0f766e, #1d4ed8);
}
.app-header h1 {
  font-family: 'Space Mono', monospace;
  font-size: 1.9rem;
  font-weight: 700;
  color: #0f172a;
  margin: 0;
}
.app-header p { color: #1f2937; margin: 8px 0 0; font-size: 0.98rem; font-weight: 600; }

/* Module banner */
.mod-banner {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 22px;
  margin-bottom: 18px;
  box-shadow: 0 3px 12px rgba(15,23,42,.04);
}
.mod-title {
  font-family: 'Space Mono', monospace;
  font-size: 1.08rem;
  font-weight: 800;
  letter-spacing: .6px;
  text-transform: uppercase;
}
.mod-title.rf  { color: #0b5f59; }
.mod-title.bnn { color: #1e40af; }
.mod-sub { color: #334155; font-size: 0.92rem; margin-top: 6px; font-weight: 600; }

/* Result boxes */
.rbox {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 16px 20px;
  border-left: 4px solid var(--amber);
  margin: 6px 0;
}
.rbox.blue   { border-left-color: var(--blue); }
.rbox.green  { border-left-color: var(--green); }
.rbox.red    { border-left-color: var(--red); }
.rbox.purple { border-left-color: var(--purple); }
.rlabel {
  font-family: 'Space Mono', monospace;
  font-size: 0.76rem;
  letter-spacing: .8px;
  text-transform: uppercase;
  color: #1f2937;
  margin-bottom: 6px;
  font-weight: 700;
}
.rvalue { font-size: 2rem; font-weight: 800; line-height: 1; color: #0f172a; }
.rvalue.amber  { color: var(--amber); }
.rvalue.blue   { color: var(--blue); }
.rvalue.green  { color: var(--green); }
.rvalue.red    { color: var(--red); }
.rvalue.purple { color: var(--purple); }
.rsub { color: #475569; font-size: 0.86rem; margin-top: 6px; font-weight: 600; }

/* Risk badges */
.rbadge {
  display: inline-block;
  padding: 5px 16px;
  border-radius: 20px;
  font-family: 'Space Mono', monospace;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 1px;
}
.r-low      { background: #ecfdf3; color: #15803d; border: 1px solid #86efac; }
.r-medium   { background: #fff7ed; color: #c2410c; border: 1px solid #fdba74; }
.r-high     { background: #fef2f2; color: #dc2626; border: 1px solid #fca5a5; }
.r-critical { background: #fee2e2; color: #b91c1c; border: 1px solid #ef4444; }

/* Notification bars */
.info-bar {
  background: #eef6f5;
  border: 1px solid #99d6cf;
  border-radius: 8px;
  padding: 11px 14px;
  font-size: 0.9rem;
  color: #0f766e;
  margin-bottom: 14px;
}
.warn-bar {
  background: #fff7ed;
  border: 1px solid #fed7aa;
  border-radius: 8px;
  padding: 11px 14px;
  font-size: 0.9rem;
  color: #9a3412;
  margin-bottom: 14px;
}

/* Native Streamlit success messages */
[data-testid="stAlert"][kind="success"],
[data-testid="stAlert"][data-baseweb="notification"] {
  background: #d1fae5 !important;
  border: 1px solid #34d399 !important;
}
[data-testid="stAlert"][kind="success"] *,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
  color: #064e3b !important;
  fill: #064e3b !important;
  opacity: 1 !important;
  font-weight: 800 !important;
}

/* Section header */
.sec-hdr {
  font-family: 'Space Mono', monospace;
  font-size: 1rem;
  letter-spacing: .9px;
  text-transform: uppercase;
  color: #0f172a;
  padding: 10px 0 9px;
  border-bottom: 2px solid var(--border);
  margin-bottom: 14px;
  font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #f1f5f9; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] label {
  color: #334155 !important;
  font-size: 0.92rem !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 700 !important;
}

/* Widget labels and radio buttons */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stRadio label,
.stRadio label p,
.stRadio label span,
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
  color: #0f172a !important;
  opacity: 1 !important;
  font-weight: 700 !important;
}
div[data-testid="stRadio"] > label,
.stTextInput label,
.stDateInput label,
.stNumberInput label,
.stTextArea label,
.stFileUploader label {
  color: #334155 !important;
  opacity: 1 !important;
  font-weight: 700 !important;
}
div[data-testid="stRadio"] [role="radiogroup"] {
  gap: 24px !important;
}

/* Expanders: no hover color shift, keep text black */
[data-testid="stExpander"] details summary,
[data-testid="stExpander"] details summary:hover,
[data-testid="stExpander"] details summary:focus,
[data-testid="stExpander"] details summary:active {
  background: #f8fafc !important;
  color: #0f172a !important;
  border-radius: 8px !important;
  transition: none !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div,
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary path,
[data-testid="stExpander"] button,
[data-testid="stExpander"] button *,
button[kind="header"],
button[kind="header"] *,
[data-testid="stExpander"] details summary *,
[data-testid="stExpander"] details summary:hover *,
[data-testid="stExpander"] details summary:focus *,
[data-testid="stExpander"] details summary:active * {
  color: #0f172a !important;
  stroke: #0f172a !important;
  fill: #0f172a !important;
  opacity: 1 !important;
  font-weight: 800 !important;
  transition: none !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] *,
section[data-testid="stSidebar"] [data-testid="stExpander"] *:hover,
section[data-testid="stSidebar"] [data-testid="stExpander"] *:focus,
section[data-testid="stSidebar"] [data-testid="stExpander"] *:active {
  color: #0f172a !important;
  fill: #0f172a !important;
  stroke: #0f172a !important;
  opacity: 1 !important;
}

/* Button */
.stButton > button {
  background: #0f766e !important;
  color: #ffffff !important;
  font-family: 'Space Mono', monospace !important;
  font-weight: 700 !important;
  font-size: 0.82rem !important;
  letter-spacing: .6px !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 10px 24px !important;
}
.stButton > button p { color: #ffffff !important; }
.stButton > button:hover {
  background: #0d5f59 !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(15,118,110,.22) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: #f8fafc;
  border-radius: 10px;
  padding: 6px;
  gap: 8px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 6px;
  color: #0f172a !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.96rem !important;
  font-weight: 700 !important;
  padding: 12px 22px !important;
  min-height: 46px !important;
  transition: none !important;
}
.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover *,
.stTabs [data-baseweb="tab"]:focus,
.stTabs [data-baseweb="tab"]:focus *,
.stTabs [data-baseweb="tab"]:active,
.stTabs [data-baseweb="tab"]:active * {
  color: #0f172a !important;
  background: transparent !important;
  opacity: 1 !important;
  transition: none !important;
}
.stTabs [aria-selected="true"] {
  background: var(--card) !important;
  color: #0f172a !important;
  border: 1px solid #0f766e !important;
  box-shadow: 0 2px 8px rgba(15,118,110,.12) !important;
}
.stTabs [aria-selected="true"] * {
  color: #0f172a !important;
}

/* Inputs */
.stTextInput input, .stDateInput input, .stNumberInput input, .stTextArea textarea {
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
  font-size: 0.96rem !important;
}
[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 8px !important;
  min-height: 42px !important;
  font-weight: 700 !important;
}

/* Metrics */
[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 18px;
}
[data-testid="stMetricLabel"] {
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  color: #1f2937 !important;
  letter-spacing: .8px;
  text-transform: uppercase;
  font-weight: 700 !important;
}
[data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700 !important; color: #0f766e !important; }
[data-testid="stMetricDelta"] {
  background: #d1fae5 !important;
  color: #064e3b !important;
  border: 1px solid #34d399 !important;
  border-radius: 999px !important;
  padding: 3px 9px !important;
  width: fit-content !important;
  font-weight: 800 !important;
}
[data-testid="stMetricDelta"] * {
  color: #064e3b !important;
  fill: #064e3b !important;
  opacity: 1 !important;
  font-weight: 800 !important;
}
[data-testid="stMetricDelta"] svg {
  width: 13px !important;
  height: 13px !important;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
header {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
}
.block-container {
  padding-top: 2rem !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
APP_DIR   = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
ARTIFACT_SEARCH_DIRS = [MODEL_DIR, APP_DIR, APP_DIR.parent]

# RF encoding maps
RAIL_SECTION_RF_MAP = {52: 1, 60: 2}
AXLE_MAP_RF         = {20: 1, 22: 2, 23: 3, 25: 4}
LINE_OPTIONS_RF     = ["DN", "UP", "SL", "OT"]
USFD_ORDER_RF       = ["GR", "OBS Joggled Plate", "OBS", "IMR"]
LRRR_OPTIONS_RF     = ["Left", "Right"]
TRACK_OPTIONS_RF    = ["BG", "LWR", "SWR"]
CURVE_OPTIONS_RF    = ["Curved", "Straight"]

# BNN encoding maps
RAIL_ORDER_BNN   = ["52KG", "60KG"]
AXLE_ORDER_BNN   = ["Up to 20.32", "22.32", "22.9", "25"]
DIVCODE_OPTIONS  = ["AGRA", "JHS", "PRYJ"]
LINE_OPTIONS_BNN = ["DN", "OTH", "SL", "UP"]
USFD_OPTIONS_BNN = ["DFW", "GR", "IMR", "OBS", "OBS Joggled Plate"]
LRRR_OPTIONS_BNN = ["Left", "Right"]
TRACK_OPTIONS_BNN = ["BG", "LWR", "SWR"]
CURVE_OPTIONS_BNN = ["Curved", "Straight"]

RAILWAY_ZONES = [
    "CR", "ER", "ECR", "ECoR", "NR", "NCR", "NER", "NFR",
    "NWR", "SR", "SCR", "SER", "SECR", "SWR", "WR", "WCR",
]

# Column alias lists for flexible DataFrame lookups
ZONE_COLUMN_ALIASES = [
    "ZONE", "RAILWAY_ZONE", "RAILWAY ZONE", "RLY_ZONE", "ZONE_CODE",
    "RAILWAY", "RLY", "RLYCODE", "RLY_CODE", "RAILWAY_CODE",
]
DIVISION_COLUMN_ALIASES = [
    "DIVCODE", "DIVISION", "DIV", "DIVISION_CODE", "RAILWAY_DIVISION",
]
SECTION_COLUMN_ALIASES = [
    "SECTION", "SEC", "ROUTE_SECTION", "SECTION_NAME", "BLOCK_SECTION",
]
DATE_COLUMN_ALIASES = [
    "DATE", "RECORD_DATE", "EVENT_DATE", "FAILURE_DATE", "FRACTURE_DATE",
    "FRACTURE_TIMESTAMP", "INSPECTION_DATE", "USFD_DATE", "LAYING_DATE",
]

# Thresholds
RISK_LOW_MAX               = 0.35
RISK_MEDIUM_MAX            = 0.65
RISK_SCORE_HORIZON_DAYS    = 1825   # 5 years  (kept for survival table display)
SURVIVAL_PLOT_POINTS       = 300

# ── Risk-score calibration ───────────────────────────────────────────────────
# The Cox model is trained on NCR rail data where rails routinely survive 10-20
# years. This pushes the ABSOLUTE survival S(t) close to 1.0 even at 5 years,
# so (1 - S(5yr)) always comes out < 0.10 – far too compressed to be useful.
#
# The *partial hazard* h = exp(Xβ) already correctly ranks relative risk, but
# it has no natural [0,1] scale. We map it with a calibrated shifted sigmoid:
#   ph = 0.5  → risk ≈ 0.20  (LOW)
#   ph = 1.0  → risk ≈ 0.35  (LOW / MEDIUM boundary – "average" rail)
#   ph = 2.0  → risk ≈ 0.54  (MEDIUM)
#   ph = 3.0  → risk ≈ 0.65  (MEDIUM / HIGH boundary)
#   ph = 5.0+ → risk ≈ 0.77+ (HIGH)
#
# Formula:  risk_score = sigmoid( k * (log(partial_hazard) - offset) )
# Derivation: anchor ph=1→0.35 and ph=3→0.65 gives k≈1.127, offset≈0.549
PARTIAL_HAZARD_SIGMOID_K      = 1.127   # steepness
PARTIAL_HAZARD_SIGMOID_OFFSET = 0.549   # log-hazard at which score = 0.50
# Fallback: when only the survival curve is available (demo mode or no Cox pkl),
# we use 2-year survival (730 days) which is more discriminating than 5-year.
RISK_FALLBACK_HORIZON_DAYS = 730    # 2 years
URGENT_REMAINING_LIFE_DAYS = 90
MIN_SELECTABLE_DATE        = pd.Timestamp("1980-01-01").date()
COX_FORMULA_COLUMNS = [
    "GMT_CARRIED",
    "RAIL_SECTION",
    "TRACK_TYPE_LWR",
    "TRACK_TYPE_SWR",
    "STRAIGHT_CURVE_STRAIGHT",
    "g_bnn",
]

PLOT_RC = {
    "figure.facecolor": "#ffffff", "axes.facecolor":  "#ffffff",
    "axes.edgecolor":   "#cbd5e1", "axes.labelcolor": "#0f172a",
    "xtick.color":      "#334155", "ytick.color":     "#334155",
    "grid.color":       "#e5e7eb", "text.color":      "#111827",
    "font.family":      "sans-serif",
    "font.weight":      "600",
    "axes.titleweight": "700",
    "axes.labelweight": "700",
}


# ──────────────────────────────────────────────────────────────────────────────
# ARTIFACT UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def resolve_artifact_path(filename: str) -> Path:
    for folder in ARTIFACT_SEARCH_DIRS:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    return MODEL_DIR / filename


def load_pickle_artifact(filename: str):
    path = resolve_artifact_path(filename)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        warnings.warn(f"Missing dependency for {filename}: {e.name}")
    except Exception as e:
        warnings.warn(f"Could not load {filename}: {e}")
    return None


def load_first_pickle_artifact(filenames: list[str]):
    for name in filenames:
        artifact = load_pickle_artifact(name)
        if artifact is not None:
            return artifact, name
    return None, ""


def is_hdf5_artifact(path: Path) -> bool:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return True
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


class H5MeanBNN:
    """Deterministic inference from tfp DenseReparameterization posterior means."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        self.layers = layers
        self.input_shape = (None, int(layers[0][0].shape[0])) if layers else (None, 0)
        self.output_shape = (None, 1)

    def predict(self, X, verbose: int = 0):
        del verbose
        out = np.asarray(X, dtype=np.float32)
        for idx, (kernel, bias) in enumerate(self.layers):
            out = out @ kernel + bias
            if idx < len(self.layers) - 1:
                out = np.maximum(out, 0.0)
        return np.asarray(out, dtype=np.float32)


def load_h5_mean_bnn(path: Path):
    try:
        import h5py
    except ModuleNotFoundError as e:
        warnings.warn(f"Missing dependency for BNN H5 fallback: {e.name}")
        return None

    layers = []
    try:
        with h5py.File(path, "r") as f:
            weight_root = f.get("model_weights")
            if weight_root is None:
                return None

            dense_names = sorted(
                [name for name in weight_root.keys() if name.startswith("dense_reparameterization")],
                key=lambda name: int(re.search(r"_(\d+)$", name).group(1)) if re.search(r"_(\d+)$", name) else 0,
            )
            for name in dense_names:
                group = weight_root[name][name]
                kernel = np.asarray(group["kernel_posterior_loc:0"], dtype=np.float32)
                bias = np.asarray(group["bias_posterior_loc:0"], dtype=np.float32)
                layers.append((kernel, bias))
    except Exception as e:
        warnings.warn(f"Could not read BNN H5 weights from {path.name}: {e}")
        return None

    return H5MeanBNN(layers) if layers else None


def load_keras_h5_artifact(filename: str):
    path = resolve_artifact_path(filename)
    if not path.exists():
        return None

    temp_path = None
    try:
        try:
            import tf_keras as keras
        except ModuleNotFoundError:
            return load_h5_mean_bnn(path)

        custom_objects = {}
        try:
            import tensorflow_probability as tfp
            custom_objects["DenseReparameterization"] = tfp.layers.DenseReparameterization
        except Exception as e:
            warnings.warn(f"TensorFlow Probability support unavailable for {filename}: {e}")

        model_path = path
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            # Keras 3 checks the extension before opening the file. This lets the
            # app read older H5 files that were accidentally named .pkl.
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
            tmp.close()
            temp_path = Path(tmp.name)
            shutil.copyfile(path, temp_path)
            model_path = temp_path

        return keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    except ModuleNotFoundError as e:
        warnings.warn(f"Missing dependency for {filename}: {e.name}. Using H5 posterior-mean fallback if possible.")
        return load_h5_mean_bnn(path)
    except Exception as e:
        warnings.warn(f"Could not load BNN model {filename} through Keras: {e}. Using H5 posterior-mean fallback if possible.")
        return load_h5_mean_bnn(path)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
    return None


def load_first_bnn_artifact(filenames: list[str]):
    for name in filenames:
        path = resolve_artifact_path(name)
        if not path.exists():
            continue

        if is_hdf5_artifact(path):
            artifact = load_keras_h5_artifact(name)
        else:
            artifact = load_pickle_artifact(name)

        if artifact is not None:
            return artifact, name
    return None, ""


def bnn_artifact_candidates(zone: str, kind: str) -> list[str]:
    """Return candidate filenames for a zone's BNN/Cox/Scaler artifact."""
    code  = normalize_zone_code(zone)
    lower = code.lower()

    if kind == "bnn":
        names = [
            f"bnn_model_{code}.h5", f"bnn_model_{lower}.h5",
            f"bnn_model_{code}.pkl", f"bnn_model_{lower}.pkl",
        ]
        if code == "NCR":
            names.extend(["bnn_model.h5", "bnn_model.pkl"])
        return names

    if kind == "scaler":
        names = [f"scaler_{code}.pkl", f"scaler_{lower}.pkl",
                 f"scaler_bnn_{code}.pkl", f"scaler_bnn_{lower}.pkl"]
        if code == "NCR":
            names.append("scaler_ncr.pkl")
        return names

    if kind == "cox":
        names = [f"cph_{code}.pkl", f"cph_{lower}.pkl",
                 f"cox_{code}.pkl",  f"cox_{lower}.pkl",
                 f"cox_model_streamlit_{code}.pkl"]
        if code == "NCR":
            names.extend(["cph_ncr.pkl", "cox_model_streamlit.pkl"])
        return names

    return []


def expected_bnn_filenames(zone: str) -> dict[str, str]:
    code = normalize_zone_code(zone)
    return {
        "BNN":    f"bnn_model_{code}.h5",
        "Cox":    f"cph_{code}.pkl",
        "Scaler": f"scaler_{code}.pkl",
    }


def zone_has_cox_artifact(zone: str) -> bool:
    return any(
        resolve_artifact_path(name).exists()
        for name in bnn_artifact_candidates(zone, "cox")
    )


# ──────────────────────────────────────────────────────────────────────────────
# ZONE UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
_ZONE_NAME_MAP = {
    "CENTRALRAILWAY":         "CR",  "EASTERNRAILWAY":          "ER",
    "EASTCENTRALRAILWAY":     "ECR", "EASTCOASTRAILWAY":        "ECoR",
    "NORTHERNRAILWAY":        "NR",  "NORTHCENTRALRAILWAY":     "NCR",
    "NORTHEASTERNRAILWAY":    "NER", "NORTHEASTFRONTIERRAILWAY":"NFR",
    "NORTHWESTERNRAILWAY":    "NWR", "SOUTHERNRAILWAY":         "SR",
    "SOUTHCENTRALRAILWAY":    "SCR", "SOUTHEASTERNRAILWAY":     "SER",
    "SOUTHEASTCENTRALRAILWAY":"SECR","SOUTHWESTERNRAILWAY":     "SWR",
    "WESTERNRAILWAY":         "WR",  "WESTCENTRALRAILWAY":      "WCR",
}


def canonical_col(name) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")


def normalize_zone_code(zone) -> str:
    text = canonical_col(zone or "NCR")
    return _ZONE_NAME_MAP.get(text, text) or "NCR"


def zone_display_name(zone) -> str:
    code = normalize_zone_code(zone)
    for option in RAILWAY_ZONES:
        if normalize_zone_code(option) == code:
            return option
    return code


# ──────────────────────────────────────────────────────────────────────────────
# RISK HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def partial_hazard_to_risk_score(partial_hazard: float) -> float:
    """Map a Cox partial hazard (>0) to a [0,1] risk score via a calibrated sigmoid.

    Uses a shifted sigmoid on the log scale so relative rankings are preserved
    while the absolute value is interpretable:

        risk = sigmoid( k * (log(ph) - offset) )

    Calibration anchors (k=1.127, offset=0.549):
        ph = 0.5  → risk ≈ 0.20  (LOW)
        ph = 1.0  → risk ≈ 0.35  (LOW / MEDIUM boundary – average rail)
        ph = 2.0  → risk ≈ 0.54  (MEDIUM)
        ph = 3.0  → risk ≈ 0.65  (MEDIUM / HIGH boundary)
        ph = 5.0  → risk ≈ 0.77  (HIGH)
    """
    if partial_hazard is None or partial_hazard <= 0:
        return 0.0
    log_ph = np.log(float(partial_hazard))
    score  = 1.0 / (1.0 + np.exp(-PARTIAL_HAZARD_SIGMOID_K * (log_ph - PARTIAL_HAZARD_SIGMOID_OFFSET)))
    return float(np.clip(score, 0.0, 1.0))


def risk_score_from_horizons(horizons: dict) -> float:
    """Fallback risk score when no partial hazard is available (demo / no Cox pkl).

    Uses 2-year survival (730 days) which is considerably more discriminating
    than the old 5-year horizon for typical NCR rail data.
    """
    # Try 2-year first; fall back to 1-year, then 5-year
    for h in [RISK_FALLBACK_HORIZON_DAYS, 365, RISK_SCORE_HORIZON_DAYS]:
        sv = horizons.get(h)
        if sv is not None:
            return float(np.clip(1.0 - float(sv), 0.0, 1.0))
    return 0.0


def risk_category_from_score(score: float) -> str:
    if score < RISK_LOW_MAX:
        return "LOW"
    if score < RISK_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def neutral_cox_covariate(cph, covariate: str, default: float = 0.0) -> float:
    norm_mean = getattr(cph, "_norm_mean", None)
    if norm_mean is not None and covariate in norm_mean.index:
        return float(norm_mean.loc[covariate])
    return float(default)


def build_cox_prediction_df(cph, cox_lin: dict, g_bnn: float) -> pd.DataFrame:
    """Build a Cox row from the active formula and align it to the loaded PKL."""
    cox_row = {**cox_lin, "g_bnn": g_bnn}
    expected_cols = list(getattr(cph, "params_", pd.Series(dtype=float)).index)
    if not expected_cols:
        expected_cols = COX_FORMULA_COLUMNS

    for col in expected_cols:
        if col not in cox_row:
            cox_row[col] = neutral_cox_covariate(cph, col)

    return pd.DataFrame([{col: cox_row[col] for col in expected_cols}])


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_rf_artifacts():
    rf_model    = load_pickle_artifact("random_forest_model.pkl")
    preprocessor = load_pickle_artifact("preprocessor.pkl")
    section_freq = load_pickle_artifact("section_freq.pkl") or {}
    if rf_model is not None and hasattr(rf_model, "n_jobs"):
        rf_model.n_jobs = 1
    return rf_model, preprocessor, section_freq


@st.cache_resource
def load_bnn_artifacts(zone: str = "NCR"):
    bnn_model, _ = load_first_bnn_artifact(bnn_artifact_candidates(zone, "bnn"))
    cph, _       = load_first_pickle_artifact(bnn_artifact_candidates(zone, "cox"))
    scaler, _    = load_first_pickle_artifact(bnn_artifact_candidates(zone, "scaler"))
    if isinstance(cph, dict):
        cph = cph.get("model")
    return bnn_model, cph, scaler


def artifact_status_rows() -> list[dict]:
    artifacts = [
        ("RF", "random_forest_model.pkl", "Random forest regressor",   "Required"),
        ("RF", "preprocessor.pkl",        "RF ColumnTransformer",       "Required"),
    ]
    for zone in RAILWAY_ZONES:
        exp = expected_bnn_filenames(zone)
        artifacts += [
            (f"BNN-Cox {zone}", exp["Cox"],    "Zone CoxPHFitter survival model",    "Required"),
            (f"BNN-Cox {zone}", exp["BNN"],    "Zone BNN non-linear covariate model","Optional"),
            (f"BNN-Cox {zone}", exp["Scaler"], "Zone BNN input scaler",              "Optional"),
        ]
    artifacts.append(("BNN-Cox NCR", "cox_model_streamlit.pkl", "Legacy NCR CoxPHFitter fallback", "Fallback"))

    rows = []
    for module, filename, description, role in artifacts:
        path  = resolve_artifact_path(filename)
        found = path.exists()
        rows.append({
            "Module":      module,
            "PKL file":    filename,
            "Role":        role,
            "Status":      "Found" if found else "Missing",
            "Size":        f"{path.stat().st_size / (1024*1024):.2f} MB" if found else "-",
            "Description": description,
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — RF
# ──────────────────────────────────────────────────────────────────────────────
def rf_lrrr_value(value) -> str:
    text = str(value or "").upper().strip()
    return "RR" if text in {"RIGHT", "R", "RR"} else "LR"


def rf_curve_value(value) -> str:
    text = str(value or "").upper().strip()
    return "STRAIGHT" if "STRAIGHT" in text or text == "ST" else "CURVE"


def rf_track_value(value) -> str:
    text = str(value or "").upper().strip()
    if "LWR" in text:
        return "LWR"
    if "SWR" in text:
        return "SWR"
    return "FP/SR"


def rf_usfd_duration(inp: dict) -> float:
    explicit = inp.get("usfd_duration")
    if explicit is not None and pd.notna(explicit):
        return max(0.0, float(explicit))
    return max(0.0, float(inp["gmt_carried"]) / max(float(inp["annual_gmt"]), 1e-6) * 365.25)


def build_rf_dataframe(inp: dict, section_freq: dict = None) -> pd.DataFrame:
    rail_enc    = RAIL_SECTION_RF_MAP.get(inp["rail_section"], 1)
    axle_enc    = AXLE_MAP_RF.get(inp["axle_load"], 2)
    annual_load = inp["annual_gmt"] * axle_enc
    speed_agmt  = inp["speed"] * inp["annual_gmt"]
    usfd_duration = rf_usfd_duration(inp)

    return pd.DataFrame([{
        "RAIL_SECTION":       rail_enc,
        "SPEED":              inp["speed"],
        "GMTCARRIED":         inp["gmt_carried"],
        "AXLELOAD":           axle_enc,
        "SERVICE_LIFE":       inp["service_life"],
        "ANNUAL_LOAD":        annual_load,
        "SPEED_AGMT":         speed_agmt,
        "USFD_DURATION":      usfd_duration,
        "SECTION":            inp["section"],
        "DIVCODE":            inp.get("divcode", "PRYJ"),
        "USFDCLASSIFICATION": inp["usfd_class"],
        "LINE":               inp["line"],
        "LRRR":               rf_lrrr_value(inp["lrrr"]),
        "TRACK_TYPE":         rf_track_value(inp["track_type"]),
        "STRAIGHT_CURVE":     rf_curve_value(inp["straight_curve"]),
        "RLYCODE":            normalize_zone_code(inp["zone"]),
    }])


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING — BNN-Cox
# ──────────────────────────────────────────────────────────────────────────────
def build_bnn_input(inp: dict):
    """
    Build the 20-element BNN input vector and Cox covariate dict.

    Feature order (matches the saved NCR scaler / BNN H5 input):
      0  ord__RAIL_SECTION          1  ord__AXLELOAD
      2  ohe__DIVCODE_JHS           3  ohe__DIVCODE_PRYJ
      4  ohe__LINE_OTH              5  ohe__LINE_SL              6  ohe__LINE_UP
      7  ohe__USFDCLASSIFICATION_OBS Joggled Plate
      8  ohe__LRRR_RR
      9  ohe__TRACK_TYPE_LWR        10  ohe__TRACK_TYPE_SWR
     11  ohe__STRAIGHT_CURVE_STRAIGHT
     12  num__ANNUAL_GMT            13  num__SPEED
     14  num__GMTCARRIED            15  num__SERVICE_LIFE
     16  num__DURATION              17  num__Kilometerage
     18  date__LAYING_DATE          19  SECTION_FREQ
    """
    rail_ord  = {"52KG": 0, "60KG": 1}.get(inp["rail_section_bnn"], -1)
    axle_ord  = {"22.32": 0, "22.9": 1}.get(inp["axle_bnn"], -1)
    div, line, usfd, lrrr = inp["divcode"], inp["line_bnn"], inp["usfd_bnn"], inp["lrrr_bnn"]
    track, curve = inp["track_bnn"], inp["straight_curve_bnn"]

    track_lwr = 1.0 if track == "LWR" else 0.0
    track_swr = 1.0 if track == "SWR" else 0.0
    straight  = 1.0 if curve == "Straight" else 0.0
    lrrr_rr   = 1.0 if str(lrrr).upper() in {"RIGHT", "RR"} else 0.0

    vec = [
        float(rail_ord), float(axle_ord),                              # 0-1  ordinal
        1.0 if div  == "JHS"               else 0.0,                  # 2
        1.0 if div  == "PRYJ"              else 0.0,                  # 3
        1.0 if line == "OTH"               else 0.0,                  # 4
        1.0 if line == "SL"                else 0.0,                  # 5
        1.0 if line == "UP"                else 0.0,                  # 6
        1.0 if usfd == "OBS Joggled Plate" else 0.0,                  # 7
        lrrr_rr,                                                       # 8
        track_lwr, track_swr,                                          # 9-10
        straight,                                                       # 11
        float(inp["annual_gmt"]),                                      # 12
        float(inp["speed"]),                                           # 13
        float(inp["gmt_carried"]),                                     # 14
        float(inp["service_life"]),                                    # 15
        float(inp.get("duration", np.nan)),                            # 16  unknown at prediction time
        float(inp.get("kilometerage", np.nan)),                        # 17
        float(inp.get("laying_date_numeric", np.nan)),                 # 18  epoch ns, unknown if absent
        float(inp.get("section_freq", np.nan)),                        # 19
    ]

    cox_lin = {
        "GMT_CARRIED":             float(inp["gmt_carried"]),
        "RAIL_SECTION":            float(rail_ord),
        "TRACK_TYPE_LWR":          track_lwr,
        "TRACK_TYPE_SWR":          track_swr,
        "STRAIGHT_CURVE_STRAIGHT": straight,
    }
    return np.array(vec, dtype=np.float32).reshape(1, -1), cox_lin


# ──────────────────────────────────────────────────────────────────────────────
# DEMO PREDICTIONS (no saved models needed)
# ──────────────────────────────────────────────────────────────────────────────
def demo_rf(inp: dict) -> dict:
    axle_enc = AXLE_MAP_RF.get(inp["axle_load"], 2)
    usfd_pen = {"GR": 0, "OBS Joggled Plate": -120, "OBS": -260, "IMR": -520}
    lrrr_pen = {"Left": 0, "Right": -80}
    gmt_frac = inp["gmt_carried"] / max(inp["annual_gmt"] * 12, 1)

    days = max(30,
        2300
        + usfd_pen.get(inp["usfd_class"], -200)
        + lrrr_pen.get(inp["lrrr"], 0)
        - inp["service_life"] * 28
        - gmt_frac * 200
        - axle_enc * 55
        + (120 if inp["track_type"] == "LWR" else 0)
        + np.random.normal(0, 55)
    )
    return {"remaining_days": int(round(days)), "remaining_years": round(days / 365, 2)}


def demo_bnn(inp: dict) -> dict:
    usfd_r   = {"DFW": 0.70, "GR": 0.05, "IMR": 0.45, "OBS": 0.35, "OBS Joggled Plate": 0.30}
    lrrr_r   = {"Left": 0.05, "Right": 0.18}
    axle_ord = AXLE_ORDER_BNN.index(inp["axle_bnn"])

    raw_hazard = (
        usfd_r.get(inp["usfd_bnn"], 0.35)
        + lrrr_r.get(inp["lrrr_bnn"], 0.10)
        + inp["service_life"] * 0.032
        + inp["gmt_carried"] / max(inp["annual_gmt"], 1) * 0.20
        + axle_ord * 0.12
        + inp["speed"] / 200.0 * 0.15
        + (0.20 if inp["track_bnn"] == "LWR" else 0.0)
    )
    t     = np.linspace(0, 5475, 400)
    scale = 4200 / (raw_hazard + 0.30)
    surv  = np.exp(-((t / scale) ** 1.45))

    horizons = {
        h: round(float(surv[int(np.argmin(np.abs(t - h)))]), 4)
        for h in [180, 365, 730, 1095, 1825, 3650]
    }
    rs  = risk_score_from_horizons(horizons)
    cat = risk_category_from_score(rs)

    return {
        "time": t.tolist(), "survival": surv.tolist(),
        "risk_score": round(rs, 4), "risk_category": cat,
        "horizons": horizons, "g_bnn": None, "partial_hazard": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION — RF
# ──────────────────────────────────────────────────────────────────────────────
def predict_rf(inp: dict) -> dict:
    rf_model, preprocessor, section_freq = load_rf_artifacts()
    if rf_model is None or preprocessor is None:
        return demo_rf(inp)

    X_df = build_rf_dataframe(inp, section_freq)
    pred = max(0.0, float(rf_model.predict(preprocessor.transform(X_df))[0]))
    return {"remaining_days": int(round(pred)), "remaining_years": round(pred / 365, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION — BNN-Cox
# ──────────────────────────────────────────────────────────────────────────────
def prepare_bnn_scaled_input(X_arr: np.ndarray, scaler) -> np.ndarray:
    X = np.array(X_arr, dtype=np.float32, copy=True)
    if scaler is None:
        return np.nan_to_num(X, nan=0.0).astype(np.float32)

    means = getattr(scaler, "mean_", None)
    expected = int(getattr(scaler, "n_features_in_", X.shape[1]))
    if X.shape[1] != expected:
        raise ValueError(f"BNN feature mismatch: app built {X.shape[1]} columns but scaler expects {expected}.")

    if means is not None and len(means) == X.shape[1]:
        missing = np.isnan(X)
        if missing.any():
            X[missing] = np.take(np.asarray(means, dtype=np.float32), np.where(missing)[1])
    else:
        X = np.nan_to_num(X, nan=0.0)

    return scaler.transform(X).astype(np.float32)


def predict_bnn_result(inp: dict) -> tuple[dict | None, str | None]:
    """Run BNN-Cox survival analysis. BNN is optional; falls back to neutral g_bnn."""
    import traceback

    zone = zone_display_name(inp.get("zone", "NCR"))
    bnn_model, cph, scaler = load_bnn_artifacts(zone)

    if cph is None:
        exp = expected_bnn_filenames(zone)
        return None, f"Cox model not found for '{zone}'. Add '{exp['Cox']}' to models/."

    try:
        X_arr, cox_lin = build_bnn_input(inp)

        # Step 1: BNN → g_bnn (use Cox training mean if the BNN is absent)
        g_bnn    = neutral_cox_covariate(cph, "g_bnn")
        bnn_mode = "Cox-only (neutral g_bnn)"
        if bnn_model is not None:
            try:
                X_s   = prepare_bnn_scaled_input(X_arr, scaler)
                g_bnn = float(bnn_model.predict(X_s, verbose=0).reshape(-1)[0])
                bnn_mode = "BNN-Cox"
            except Exception as e:
                warnings.warn(f"BNN inference failed; using neutral g_bnn: {e}")
                g_bnn = neutral_cox_covariate(cph, "g_bnn")
                bnn_mode = "Cox-only (BNN error, neutral g_bnn)"

        # Step 2: Cox survival prediction
        cox_df = build_cox_prediction_df(cph, cox_lin, g_bnn)

        partial_hz = float(cph.predict_partial_hazard(cox_df).values[0])
        t_grid     = np.linspace(1, 5475, 300)
        s_vals     = cph.predict_survival_function(cox_df, times=t_grid).iloc[:, 0].values

        horizons = {
            h: round(float(s_vals[int(np.argmin(np.abs(t_grid - h)))]), 4)
            for h in [180, 365, 730, 1095, 1825, 3650]
        }
        # Use sigmoid-mapped partial hazard for the primary risk score.
        # This avoids the near-zero scores produced by (1 - S(5yr)) when the
        # baseline survival is very high (typical for long-lived NCR rails).
        rs  = partial_hazard_to_risk_score(partial_hz)
        cat = risk_category_from_score(rs)

        return dict(
            time=t_grid.tolist(), survival=s_vals.tolist(),
            risk_score=round(rs, 4), risk_category=cat,
            horizons=horizons, g_bnn=round(g_bnn, 4),
            partial_hazard=round(partial_hz, 4),
            model_zone=zone, model_mode=bnn_mode,
        ), None

    except Exception as e:
        traceback.print_exc()
        return None, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# ROW PARSING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def row_lookup(row: pd.Series, aliases: list[str], default=None):
    canonical = {canonical_col(col): col for col in row.index}
    for alias in aliases:
        col = canonical.get(canonical_col(alias))
        if col is not None:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                return val
    return default


def parse_float(value, default: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float(default)
    match = re.search(r"-?\d+(\.\d+)?", str(value).strip().replace(",", ""))
    return float(match.group(0)) if match else float(default)


def parse_choice(value, options: list[str], default: str) -> str:
    text = str(value or "").strip()
    if not text:
        return default
    by_upper = {opt.upper(): opt for opt in options}
    return by_upper.get(text.upper(), default)


def parse_rail_section(value, default: int) -> int:
    num = int(round(parse_float(value, default)))
    return 60 if num >= 56 else 52


def parse_axle_load(value, default: int) -> int:
    num = parse_float(value, default)
    return min(AXLE_MAP_RF, key=lambda opt: abs(opt - num))


def parse_line_rf(value, default: str) -> str:
    text = str(value or "").upper().strip()
    if text in LINE_OPTIONS_RF:
        return text
    if text in {"DOWN", "D"}:
        return "DN"
    if text in {"SINGLE", "SL"}:
        return "SL"
    return default if default in LINE_OPTIONS_RF else "OT"


def parse_usfd_rf(value, default: str) -> str:
    text = str(value or "").upper().strip()
    if "JOGGLED" in text:
        return "OBS Joggled Plate"
    if "IMR"  in text:
        return "IMR"
    if "OBS"  in text:
        return "OBS"
    if "GR"   in text or "DFW" in text:
        return "GR"
    return default if default in USFD_ORDER_RF else "GR"


def parse_lrrr(value, default: str) -> str:
    text = str(value or "").upper().strip()
    if text.startswith("L"):
        return "Left"
    if text.startswith("R"):
        return "Right"
    return default if default in LRRR_OPTIONS_RF else "Left"


def parse_track(value, default: str) -> str:
    text = str(value or "").upper().strip()
    for opt in ["LWR", "SWR", "BG"]:
        if opt in text:
            return opt
    return default if default in TRACK_OPTIONS_RF else "BG"


def parse_curve(value, default: str) -> str:
    text = str(value or "").upper().strip()
    if "CURV" in text:
        return "Curved"
    if "STRAIGHT" in text or text == "ST":
        return "Straight"
    return default if default in CURVE_OPTIONS_RF else "Straight"


def parse_usfd_duration_row(row: pd.Series):
    explicit = row_lookup(row, ["USFD_DURATION", "USFD DURATION", "USFD_AGE", "DAYS_SINCE_USFD"], None)
    if explicit is not None:
        parsed = parse_float(explicit, np.nan)
        if pd.notna(parsed):
            return parsed

    usfd_raw = row_lookup(row, ["USFDTIMESTAMP", "USFD_TIMESTAMP", "USFDTESTDATE", "USFD_DATE"], None)
    laying_raw = row_lookup(row, ["LAYING_DATE", "LAYINGDATE", "LAYINGDATE_1", "LAYINGDATE.1"], None)
    if usfd_raw is not None and laying_raw is not None:
        usfd_date = pd.to_datetime(usfd_raw, errors="coerce", dayfirst=True)
        laying_date = pd.to_datetime(laying_raw, errors="coerce", dayfirst=True)
        if pd.notna(usfd_date) and pd.notna(laying_date):
            duration = (usfd_date - laying_date).days
            if duration > 0:
                return float(duration)
    return None


def api_row_to_rf_input(row: pd.Series, base_inp: dict) -> dict:
    """Parse a DataFrame row into an RF input dict, falling back to base_inp values."""
    rail_section = parse_rail_section(
        row_lookup(row, ["RAIL_SECTION", "RAIL SECTION", "RAILSECTION"], base_inp["rail_section"]),
        base_inp["rail_section"],
    )
    axle_load = parse_axle_load(
        row_lookup(row, ["AXLELOAD", "AXLE_LOAD", "AXLE LOAD"], base_inp["axle_load"]),
        base_inp["axle_load"],
    )
    return {
        **base_inp,
        "zone":         zone_display_name(row_lookup(row, ["RLYCODE"], base_inp["zone"])),
        "section":      str(row_lookup(row, SECTION_COLUMN_ALIASES, base_inp["section"])),
        "rail_section": rail_section,
        "rail_section_bnn": "52KG" if rail_section == 52 else "60KG",
        "track_type":   parse_track(row_lookup(row, ["TRACK_TYPE", "TRACK"], base_inp["track_type"]), base_inp["track_type"]),
        "straight_curve": parse_curve(row_lookup(row, ["STRAIGHT_CURVE", "ALIGNMENT"], base_inp["straight_curve"]), base_inp["straight_curve"]),
        "service_life": parse_float(row_lookup(row, ["SERVICE_LIFE", "AGE"], base_inp["service_life"]), base_inp["service_life"]),
        "annual_gmt":   parse_float(row_lookup(row, ["ANNUAL_GMT", "GMT_PER_YEAR"], base_inp["annual_gmt"]), base_inp["annual_gmt"]),
        "gmt_carried":  parse_float(row_lookup(row, ["GMTCARRIED", "GMT_CARRIED"], base_inp["gmt_carried"]), base_inp["gmt_carried"]),
        "speed":        parse_float(row_lookup(row, ["SPEED", "MAX_SPEED"], base_inp["speed"]), base_inp["speed"]),
        "usfd_duration": parse_usfd_duration_row(row),
        "axle_load":    axle_load,
        "axle_bnn":     {20: "Up to 20.32", 22: "22.32", 23: "22.9", 25: "25"}[axle_load],
        "usfd_class":   parse_usfd_rf(row_lookup(row, ["USFD_CLASSIFICATION", "USFDCLASSIFICATION", "USFD"], base_inp["usfd_class"]), base_inp["usfd_class"]),
        "lrrr":         parse_lrrr(row_lookup(row, ["LRRR"], base_inp["lrrr"]), base_inp["lrrr"]),
        "line":         parse_line_rf(row_lookup(row, ["LINE"], base_inp["line"]), base_inp["line"]),
        "divcode":      str(row_lookup(row, DIVISION_COLUMN_ALIASES, base_inp.get("divcode", "PRYJ")) or "PRYJ").strip().upper(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# BATCH PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def predict_remaining_life_rows(
    df: pd.DataFrame,
    base_inp: dict,
    date_col: str = "",
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    rf_model, preprocessor, section_freq = load_rf_artifacts()
    model_ok = rf_model is not None and preprocessor is not None
    rows, errors = [], []

    for idx, record in (df.head(max_rows) if max_rows else df).iterrows():
        try:
            inp = api_row_to_rf_input(record, base_inp)
            if model_ok:
                X_t  = preprocessor.transform(build_rf_dataframe(inp, section_freq))
                days = max(0.0, float(rf_model.predict(X_t)[0]))
            else:
                days = float(demo_rf(inp)["remaining_days"])

            row = {
                "Source Row":          idx,
                "Zone":                inp["zone"],
                "Section":             inp["section"],
                "Track Type":          inp["track_type"],
                "Alignment":           inp["straight_curve"],
                "USFD":                inp["usfd_class"],
                "LRRR":                inp["lrrr"],
                "Remaining Life Days": int(round(days)),
                "Remaining Life Years":round(days / 365, 2),
            }
            if date_col and date_col in record.index:
                row["Record Date"] = record[date_col]
            rows.append(row)
        except Exception as e:
            errors.append(f"Row {idx}: {e}")

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("Remaining Life Days").reset_index(drop=True)
    return result, errors


def predict_api_remaining_life(
    df: pd.DataFrame,
    base_inp: dict,
    threshold_days: int,
    date_col: str = "",
) -> tuple[pd.DataFrame, list[str]]:
    """Return only rows with predicted remaining life ≤ threshold_days."""
    result, errors = predict_remaining_life_rows(df, base_inp, date_col)
    if result.empty:
        return result, errors
    result = result.sort_values("Remaining Life Days").reset_index(drop=True)
    urgent = result[result["Remaining Life Days"] <= threshold_days].copy()
    return urgent, errors


def predict_survival_rows(
    df: pd.DataFrame,
    base_inp: dict,
    max_rows: int,
    date_col: str = "",
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    rows, all_curves, errors = [], [], []

    for idx, record in df.head(max_rows).iterrows():
        try:
            row_inp = api_row_to_rf_input(record, base_inp)
            div_val = row_lookup(record, DIVISION_COLUMN_ALIASES, base_inp.get("divcode", "PRYJ"))
            row_inp["divcode"]          = str(div_val or "PRYJ").strip().upper()
            row_inp["usfd_bnn"]         = parse_choice(row_lookup(record, ["USFD_CLASSIFICATION", "USFD"], base_inp["usfd_bnn"]), USFD_OPTIONS_BNN, base_inp["usfd_bnn"])
            row_inp["lrrr_bnn"]         = row_inp["lrrr"]
            row_inp["line_bnn"]         = parse_choice(row_lookup(record, ["LINE"], base_inp["line_bnn"]), LINE_OPTIONS_BNN, base_inp["line_bnn"])
            row_inp["track_bnn"]        = row_inp["track_type"]
            row_inp["straight_curve_bnn"] = row_inp["straight_curve"]

            result, err = predict_bnn_result(row_inp)
            if err or result is None:
                errors.append(f"Row {idx}: {err}")
                continue

            location = {
                "Railway Zone": str(row_lookup(record, ["RLYCODE"], row_inp["zone"]) or "-"),
                "Division":     str(row_lookup(record, DIVISION_COLUMN_ALIASES, row_inp.get("divcode", "-")) or "-"),
                "Section":      str(row_lookup(record, SECTION_COLUMN_ALIASES, row_inp["section"]) or "-"),
            }
            label = " | ".join(v for v in location.values() if v and v != "-") or f"row {idx}"
            result["_label"]         = label
            result["_risk_category"] = result["risk_category"]
            all_curves.append(result)

            out = {
                "Source Row": idx, **location,
                "BNN Model Zone":  result.get("model_zone", row_inp["zone"]),
                "Model Mode":      result.get("model_mode", "PKL"),
                "5-Year Risk Score": result["risk_score"],
                "Risk Category":   result["risk_category"],
                "1-Year Survival": result["horizons"].get(365),
                "5-Year Survival": result["horizons"].get(1825),
            }
            if date_col and date_col in record.index:
                out["Record Date"] = record[date_col]
            rows.append(out)
        except Exception as e:
            errors.append(f"Row {idx}: {e}")

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("5-Year Risk Score", ascending=False).reset_index(drop=True)
    return result_df, all_curves, errors


# ──────────────────────────────────────────────────────────────────────────────
# API FETCH
# ──────────────────────────────────────────────────────────────────────────────
def fetch_api_dataframe(
    endpoint: str,
    username: str,
    password: str,
    from_date,
    to_date,
) -> pd.DataFrame:
    if not username.strip() or not password:
        raise RuntimeError("Enter both IRCEP username and password before fetching.")
    if from_date is None or to_date is None:
        raise RuntimeError("Select both From and To dates before fetching.")

    endpoint = endpoint.strip() or "https://ircep.gov.in/TMSREST/IITDP4Controller"
    params   = {
        "param":    "RAILMASTER",
        "fromdate": from_date.strftime("%d/%m/%Y"),
        "todate":   to_date.strftime("%d/%m/%Y"),
    }

    try:
        session = requests.Session()
        session.trust_env = False
        response = session.post(
            endpoint, params=params,
            auth=HTTPBasicAuth(username.strip(), password),
            timeout=700,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise RuntimeError("Request timed out. Please try again.") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API error: {e}") from e

    if not response.text.strip():
        raise RuntimeError("Server returned an empty response. Check credentials and IRCEP access.")

    try:
        payload = response.json()
    except ValueError as e:
        preview = re.sub(r"\s+", " ", response.text[:300]).strip()
        raise RuntimeError(f"API did not return valid JSON. Preview: {preview!r}") from e

    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        for key in ("data", "Data", "result", "records"):
            val = payload.get(key)
            if isinstance(val, list):
                return pd.json_normalize(val)
        return pd.json_normalize([payload])
    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# DATE FILTER HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def find_date_columns(df: pd.DataFrame) -> list[str]:
    alias_set = {canonical_col(a) for a in DATE_COLUMN_ALIASES}
    candidates = []
    for col in df.columns:
        canon = canonical_col(col)
        is_date_name  = canon in alias_set or "DATE" in canon or "TIMESTAMP" in canon
        is_date_dtype = pd.api.types.is_datetime64_any_dtype(df[col])
        if (is_date_name or is_date_dtype) and pd.to_datetime(df[col], errors="coerce", dayfirst=True).notna().any():
            candidates.append(str(col))
    return candidates


def apply_date_range_filter(df: pd.DataFrame, date_col: str, start_date, end_date) -> tuple[pd.DataFrame, int, int]:
    dates     = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    start     = pd.Timestamp(start_date)
    end       = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    mask      = dates.between(start, end, inclusive="both")
    return df.loc[mask].copy(), int(mask.sum()), int(dates.isna().sum())


def unique_date_options(df: pd.DataFrame, date_col: str) -> list:
    dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dropna().dt.date
    return sorted(pd.unique(dates).tolist())


def ensure_selectbox_state_in_options(key: str, options: list) -> None:
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state.pop(key, None)


def filter_loaded_data_by_file_date_options(df: pd.DataFrame, key_prefix: str, title: str = "Date Range") -> tuple[pd.DataFrame, dict]:
    meta = {"enabled": False, "date_col": "", "start": None, "end": None,
            "invalid_count": 0, "original_count": len(df), "filtered_count": len(df)}
    date_cols = find_date_columns(df)
    if not date_cols:
        st.markdown("<div class='warn-bar'>No date column found; date filter not applied.</div>", unsafe_allow_html=True)
        return df, meta

    with st.expander(title, expanded=True):
        date_col = st.selectbox("Date column", date_cols, key=f"{key_prefix}_date_column")
        date_options = unique_date_options(df, date_col)

        if not date_options:
            st.markdown(f"<div class='warn-bar'>Column <b>{date_col}</b> could not be parsed.</div>", unsafe_allow_html=True)
            return df.iloc[0:0].copy(), {**meta, "enabled": True, "date_col": date_col, "filtered_count": 0}

        from_key = f"{key_prefix}_date_from_option"
        to_key   = f"{key_prefix}_date_to_option"
        ensure_selectbox_state_in_options(from_key, date_options)
        ensure_selectbox_state_in_options(to_key, date_options)

        col_from, col_to = st.columns(2)
        with col_from:
            start_date = st.selectbox(
                "From",
                date_options,
                index=0,
                format_func=lambda d: pd.Timestamp(d).strftime("%Y/%m/%d"),
                key=from_key,
            )
        with col_to:
            end_date = st.selectbox(
                "To",
                date_options,
                index=len(date_options) - 1,
                format_func=lambda d: pd.Timestamp(d).strftime("%Y/%m/%d"),
                key=to_key,
            )

    if start_date > end_date:
        st.markdown("<div class='warn-bar'>From date is after To date.</div>", unsafe_allow_html=True)
        return df.iloc[0:0].copy(), {**meta, "enabled": True, "date_col": date_col,
                                     "start": start_date, "end": end_date, "filtered_count": 0}

    filtered, filtered_count, invalid_count = apply_date_range_filter(df, date_col, start_date, end_date)
    st.markdown(
        f"<div class='info-bar'>Date filter: <b>{filtered_count:,}</b> of <b>{len(df):,}</b> records kept "
        f"({start_date} to {end_date}) using <b>{date_col}</b>.</div>",
        unsafe_allow_html=True,
    )
    if invalid_count:
        st.caption(f"{invalid_count:,} record(s) had unreadable dates and were excluded.")
    return filtered, {**meta, "enabled": True, "date_col": date_col, "start": start_date, "end": end_date,
                      "invalid_count": invalid_count, "filtered_count": filtered_count}


def filter_loaded_data_by_selected_dates(df: pd.DataFrame, key_prefix: str, start_date, end_date) -> tuple[pd.DataFrame, dict]:
    meta = {"enabled": True, "date_col": "", "start": start_date, "end": end_date,
            "invalid_count": 0, "original_count": len(df), "filtered_count": len(df)}
    date_cols = find_date_columns(df)
    if not date_cols:
        st.markdown("<div class='warn-bar'>No date column found; date filter not applied.</div>", unsafe_allow_html=True)
        return df, {**meta, "enabled": False}

    date_col = st.selectbox("Date column", date_cols, key=f"{key_prefix}_date_column")
    if start_date > end_date:
        st.markdown("<div class='warn-bar'>From date is after To date.</div>", unsafe_allow_html=True)
        return df.iloc[0:0].copy(), {**meta, "date_col": date_col, "filtered_count": 0}

    filtered, filtered_count, invalid_count = apply_date_range_filter(df, date_col, start_date, end_date)
    st.markdown(
        f"<div class='info-bar'>Date filter: <b>{filtered_count:,}</b> of <b>{len(df):,}</b> records kept "
        f"({start_date} → {end_date}) using <b>{date_col}</b>.</div>",
        unsafe_allow_html=True,
    )
    return filtered, {**meta, "date_col": date_col, "invalid_count": invalid_count, "filtered_count": filtered_count}


def render_date_filter(df: pd.DataFrame, key_prefix: str) -> tuple[pd.DataFrame, dict]:
    meta      = {"enabled": False, "date_col": "", "start": None, "end": None,
                 "invalid_count": 0, "original_count": len(df), "filtered_count": len(df)}
    date_cols = find_date_columns(df)
    if not date_cols:
        st.markdown("<div class='warn-bar'>No date column found; date filtering unavailable.</div>", unsafe_allow_html=True)
        return df, meta

    st.markdown("<div class='sec-hdr'>Date Range Filter</div>", unsafe_allow_html=True)
    if not st.checkbox("Filter records by date", value=True, key=f"{key_prefix}_date_filter_enabled"):
        return df, meta

    col_date, col_from, col_to = st.columns([2, 1, 1])
    with col_date:
        date_col = st.selectbox("Date column", date_cols, key=f"{key_prefix}_date_column")

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dropna()
    if parsed_dates.empty:
        st.markdown(f"<div class='warn-bar'>Column <b>{date_col}</b> could not be parsed.</div>", unsafe_allow_html=True)
        return df.iloc[0:0].copy(), {**meta, "enabled": True, "date_col": date_col, "filtered_count": 0}

    min_d, max_d = parsed_dates.min().date(), parsed_dates.max().date()
    with col_from:
        start_date = st.date_input("From date", value=min_d, min_value=MIN_SELECTABLE_DATE, max_value=max_d, key=f"{key_prefix}_date_from")
    with col_to:
        end_date   = st.date_input("To date",   value=max_d, min_value=MIN_SELECTABLE_DATE, max_value=max_d, key=f"{key_prefix}_date_to")

    if start_date > end_date:
        st.markdown("<div class='warn-bar'>From date is after To date.</div>", unsafe_allow_html=True)
        return df.iloc[0:0].copy(), {**meta, "enabled": True, "date_col": date_col, "start": start_date, "end": end_date, "filtered_count": 0}

    filtered, filtered_count, invalid_count = apply_date_range_filter(df, date_col, start_date, end_date)
    st.markdown(
        f"<div class='info-bar'>Date filter: <b>{filtered_count:,}</b> of <b>{len(df):,}</b> records kept.</div>",
        unsafe_allow_html=True,
    )
    if invalid_count:
        st.caption(f"{invalid_count:,} record(s) had unreadable dates and were excluded.")
    return filtered, {**meta, "enabled": True, "date_col": date_col, "start": start_date, "end": end_date,
                      "invalid_count": invalid_count, "filtered_count": filtered_count}


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def plot_gauge(days: int, max_days: int = 5475):
    plt.rcParams.update(PLOT_RC)
    fig, ax = plt.subplots(figsize=(9, 2.2), facecolor="#ffffff")
    frac = min(days / max_days, 1.0)
    col  = "#0f766e" if frac > 0.5 else ("#d97706" if frac > 0.25 else "#dc2626")

    ax.barh([0], [max_days], height=0.42, color="#e5e7eb", edgecolor="none")
    ax.barh([0], [days],     height=0.55, color=col,       edgecolor="none")
    for pct, lbl in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%")]:
        ax.axvline(max_days * pct, color="#cbd5e1", lw=1, ls="--", alpha=0.85)
        ax.text(max_days * pct, -0.65, lbl, ha="center", fontsize=8.5, color="#334155", fontweight="700")
    ax.axvline(days, color="#111827", lw=1.4, alpha=0.85)
    ax.text(min(days + max_days * 0.015, max_days * 0.97), 0.52,
            f"{days:,} d", color="#0f172a", fontsize=11, fontweight="bold", va="bottom")
    ax.set_xlim(0, max_days); ax.set_ylim(-0.9, 0.9); ax.set_yticks([])
    ax.set_xlabel("Remaining Days", fontsize=10.5, labelpad=8, fontweight="700", color="#0f172a")
    ax.set_title("Remaining Service Life", fontsize=12, fontweight="bold", color="#0f172a", pad=10)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout(pad=1)
    return fig


def plot_survival(result: dict, max_points: int | None = None):
    plt.rcParams.update(PLOT_RC)
    fig = plt.figure(figsize=(13, 5.2), facecolor="#ffffff")
    gs  = GridSpec(1, 2, fig, width_ratios=[2.3, 1], wspace=0.38)
    t   = np.array(result["time"])
    s   = np.array(result["survival"])
    yrs = t / 365

    if max_points and len(t) > max_points:
        keep = np.unique(np.linspace(0, len(t) - 1, int(max_points)).astype(int))
        yrs_plot, s_plot = yrs[keep], s[keep]
    else:
        yrs_plot, s_plot = yrs, s

    # Survival curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(yrs_plot, s_plot, color="#0f766e", lw=2.2, zorder=5)
    ax1.axhline(0.5, color="#9ca3af", lw=1, ls="--", alpha=0.8)
    ax1.text(max(yrs) * 0.985, 0.51, "S=0.50", color="#334155", fontsize=9, ha="right", fontweight="700")

    h_cols   = ["#0f766e","#1d4ed8","#6d28d9","#d97706","#dc2626","#be123c"]
    h_lbls   = ["6M","1Y","2Y","3Y","5Y","10Y"]
    offsets  = [(6,14),(6,28),(6,42),(6,24),(8,10),(8,-14)]
    for i, ((h_d, sv), col, lbl) in enumerate(zip(result["horizons"].items(), h_cols, h_lbls)):
        hyr = h_d / 365
        if hyr <= max(yrs) + 0.1:
            idx = int(np.argmin(np.abs(t - h_d)))
            ax1.scatter([yrs[idx]], [s[idx]], color=col, s=42, zorder=8, edgecolors="#ffffff", lw=0.7)
            ax1.annotate(f"{lbl}: {sv:.0%}", (yrs[idx], s[idx]),
                         textcoords="offset points", xytext=offsets[i],
                         fontsize=8.5, color=col, fontweight="700",
                         arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.75))

    ax1.set_xlim(0, max(yrs)); ax1.set_ylim(-0.04, 1.06)
    ax1.set_xlabel("Time (Years)", fontsize=10.5, labelpad=8, fontweight="700", color="#0f172a")
    ax1.set_ylabel("Survival Probability  S(t)", fontsize=10.5, labelpad=8, fontweight="700", color="#0f172a")
    ax1.set_title("BNN-Cox Survival Function", fontsize=12, fontweight="bold", color="#0f172a", pad=12)
    ax1.grid(True, alpha=1, lw=0.7); ax1.tick_params(labelsize=9, colors="#334155")
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    # Horizon bar chart
    ax2     = fig.add_subplot(gs[1])
    h_vals  = list(result["horizons"].values())
    h_names = h_lbls[:len(h_vals)]
    bcols   = ["#0f766e" if v > 0.65 else ("#d97706" if v > 0.35 else "#dc2626") for v in h_vals]
    bars    = ax2.barh(h_names, h_vals, color=bcols, height=0.52, edgecolor="none", zorder=3)
    for bar, val in zip(bars, h_vals):
        ax2.text(min(val + 0.02, 0.93), bar.get_y() + bar.get_height() / 2,
                 f"{val:.1%}", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("Survival Probability", fontsize=10.5, fontweight="700", color="#0f172a")
    ax2.set_title("Key Horizons", fontsize=12, fontweight="bold", color="#0f172a", pad=12)
    ax2.axvline(0.5, color="#9ca3af", lw=1, ls="--", alpha=0.75)
    ax2.grid(True, axis="x", alpha=1, lw=0.7); ax2.tick_params(labelsize=9.5, colors="#334155")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    plt.tight_layout(pad=1.5)
    return fig


def plot_survival_all(all_curves: list[dict], max_points: int | None = None):
    from matplotlib.lines import Line2D
    plt.rcParams.update(PLOT_RC)

    RISK_PALETTE = {"LOW": "#0f766e", "MEDIUM": "#d97706", "HIGH": "#dc2626"}
    fig = plt.figure(figsize=(13, 5.5), facecolor="#ffffff")
    gs  = GridSpec(1, 2, fig, width_ratios=[2.3, 1], wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    sorted_curves = sorted(all_curves, key=lambda r: r["risk_score"])
    show_legend   = len(sorted_curves) <= 20

    for curve in sorted_curves:
        t, s = np.array(curve["time"]), np.array(curve["survival"])
        yrs  = t / 365
        if max_points and len(t) > max_points:
            keep = np.unique(np.linspace(0, len(t) - 1, int(max_points)).astype(int))
            yrs, s = yrs[keep], s[keep]
        cat   = curve.get("_risk_category", "LOW")
        col   = RISK_PALETTE.get(cat, "#334155")
        label = curve.get("_label", "") if show_legend else None
        ax1.plot(yrs, s, color=col, lw=1.4, alpha=0.72, label=label)

    ax1.axhline(0.5, color="#9ca3af", lw=1, ls="--", alpha=0.8)
    ax1.text(max(np.array(all_curves[0]["time"]) / 365) * 0.985, 0.51,
             "S=0.50", color="#334155", fontsize=9, ha="right", fontweight="700")
    ax1.set_xlim(left=0); ax1.set_ylim(-0.04, 1.06)
    ax1.set_xlabel("Time (Years)", fontsize=10.5, labelpad=8, fontweight="700", color="#0f172a")
    ax1.set_ylabel("Survival Probability  S(t)", fontsize=10.5, labelpad=8, fontweight="700", color="#0f172a")
    n = len(sorted_curves)
    ax1.set_title(f"BNN-Cox Survival Curves — {n} Record{'s' if n != 1 else ''}",
                  fontsize=12, fontweight="bold", color="#0f172a", pad=12)
    ax1.grid(True, alpha=0.45, lw=0.7); ax1.tick_params(labelsize=9, colors="#334155")
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    if show_legend:
        ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.9, edgecolor="#d1d5db")
    else:
        cats_present = sorted({c.get("_risk_category", "LOW") for c in all_curves},
                              key=lambda x: ["LOW", "MEDIUM", "HIGH"].index(x))
        handles = [Line2D([0], [0], color=RISK_PALETTE[cat], lw=2, label=cat) for cat in cats_present]
        ax1.legend(handles=handles, title="Risk Category", fontsize=9, title_fontsize=8.5,
                   loc="upper right", framealpha=0.9, edgecolor="#d1d5db")

    # Risk category bar chart
    cat_order  = ["LOW", "MEDIUM", "HIGH"]
    cat_counts = {cat: sum(1 for c in all_curves if c.get("_risk_category") == cat) for cat in cat_order}
    cats   = [c for c in cat_order if cat_counts[c] > 0]
    counts = [cat_counts[c] for c in cats]
    bars   = ax2.barh(cats, counts, color=[RISK_PALETTE[c] for c in cats], height=0.52, edgecolor="none", zorder=3)
    for bar, cnt in zip(bars, counts):
        ax2.text(cnt + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(cnt), va="center", fontsize=10, fontweight="bold", color="#0f172a")
    ax2.set_xlabel("Number of Records", fontsize=10.5, fontweight="700", color="#0f172a")
    ax2.set_title("Risk Category\nBreakdown", fontsize=12, fontweight="bold", color="#0f172a", pad=12)
    ax2.set_xlim(0, max(counts) * 1.25 if counts else 1)
    ax2.grid(True, axis="x", alpha=0.45, lw=0.7); ax2.tick_params(labelsize=9.5, colors="#334155")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    plt.tight_layout(pad=1.5)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# DATA SOURCE PANEL
# ──────────────────────────────────────────────────────────────────────────────
def clamp_date(value, min_date, max_date):
    value = pd.Timestamp(value).date()
    if value < min_date:
        return min_date
    if value > max_date:
        return max_date
    return value


def clear_invalid_selectbox_value(key: str, options: list) -> None:
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state.pop(key, None)


def render_full_year_date_selector(label: str, default_date, key_prefix: str, min_date, max_date):
    default_date = clamp_date(default_date, min_date, max_date)
    return st.date_input(
        label,
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        format="YYYY/MM/DD",
        key=f"{key_prefix}_calendar",
    )


def render_manual_date_range(key_prefix: str, title: str = "Date Range") -> tuple:
    today = pd.Timestamp.today().date()
    default_from = (pd.Timestamp.today() - pd.DateOffset(months=3)).date()
    with st.expander(title, expanded=True):
        col_from, col_to = st.columns(2)
        with col_from:
            start = render_full_year_date_selector("From", default_from, f"{key_prefix}_manual_from", MIN_SELECTABLE_DATE, today)
        with col_to:
            end = render_full_year_date_selector("To", today, f"{key_prefix}_manual_to", MIN_SELECTABLE_DATE, today)
    return start, end


def render_data_source() -> dict:
    st.markdown("<div class='sec-hdr'>Data Source</div>", unsafe_allow_html=True)
    source  = st.radio("Input", ["Live API (IRCEP)", "Upload Excel/CSV", "Paste JSON"], horizontal=True, key="data_source")
    payload = {"source": source, "data": None, "endpoint": "", "date_filter": {}}

    if source == "Live API (IRCEP)":
        today       = pd.Timestamp.today().date()
        default_from = (pd.Timestamp.today() - pd.DateOffset(months=3)).date()

        with st.expander("Credentials & Date Range", expanded=True):
            endpoint   = st.text_input("IRCEP API endpoint", value="https://ircep.gov.in/TMSREST/IITDP4Controller", key="ircep_api_endpoint")
            col1, col2 = st.columns(2)
            with col1:
                username = st.text_input("Username", value="IITDP4", key="ircep_api_username")
                password = st.text_input("Password", value="",       type="password", key="ircep_api_password")
            with col2:
                api_from = st.date_input("From", value=default_from, min_value=MIN_SELECTABLE_DATE, key="ircep_api_from")
                api_to   = st.date_input("To",   value=today,        min_value=MIN_SELECTABLE_DATE, key="ircep_api_to")

            fetch_col, clear_col, _ = st.columns([1, 1, 4])
            with fetch_col:
                fetch_api = st.button("Fetch", key="fetch_api_data")
            with clear_col:
                clear_api = st.button("Clear", key="clear_api_data")

        payload["endpoint"]    = endpoint
        payload["date_filter"] = {"enabled": True, "date_col": "", "start": api_from, "end": api_to,
                                   "invalid_count": 0, "original_count": 0, "filtered_count": 0}

        if clear_api:
            for key in ["api_records_df", "api_records_signature", "api_remaining_life_alerts", "api_remaining_life_signature"]:
                st.session_state.pop(key, None)

        if fetch_api:
            if not endpoint.strip():
                st.error("Enter the API endpoint before fetching.")
            elif api_from > api_to:
                st.error("From date must be before or equal to To date.")
            else:
                with st.spinner("Fetching records from API..."):
                    try:
                        df = fetch_api_dataframe(endpoint, username, password, api_from, api_to)
                        api_sig = (endpoint, username, str(api_from), str(api_to))
                        st.session_state["api_records_df"]        = df
                        st.session_state["api_records_signature"] = api_sig
                        st.session_state.pop("api_remaining_life_alerts", None)
                        payload["data"] = df
                        payload["date_filter"].update({"original_count": len(df), "filtered_count": len(df)})
                        st.success(f"Fetched {len(df):,} record(s) from {api_from} to {api_to}")
                    except Exception as e:
                        st.error(f"API fetch failed: {e}")

        cached_df  = st.session_state.get("api_records_df")
        cached_sig = st.session_state.get("api_records_signature")
        api_sig    = (endpoint, username, str(api_from), str(api_to))
        if cached_df is not None and cached_sig == api_sig:
            payload["data"] = cached_df
            payload["date_filter"].update({"original_count": len(cached_df), "filtered_count": len(cached_df)})
            st.dataframe(cached_df.head(20), use_container_width=True)

        st.markdown("<div class='info-bar'>Live API mode: records fetched for the selected date range.</div>", unsafe_allow_html=True)

    elif source == "Upload Excel/CSV":
        upload_from, upload_to = render_manual_date_range("upload", "Upload Date Range")
        uploaded = st.file_uploader("Upload Excel/CSV", type=["csv", "xlsx", "xls"], key="source_upload")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
                st.success(f"Loaded {len(df):,} rows from {uploaded.name}")
                filtered_df, date_filter = filter_loaded_data_by_selected_dates(df, "upload", upload_from, upload_to)
                payload["data"]        = filtered_df
                payload["date_filter"] = date_filter
                st.dataframe(filtered_df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read file: {e}")

    else:  # Paste JSON
        json_from, json_to = render_manual_date_range("json", "JSON Date Range")
        raw_json = st.text_area("Paste JSON", height=150,
                                placeholder='[{"zone":"NCR","section":"ALD-MGS","rail_section":60}]',
                                key="source_json")
        if raw_json.strip():
            try:
                parsed = json.loads(raw_json)
                df     = pd.json_normalize(parsed if isinstance(parsed, list) else [parsed])
                st.success(f"Parsed {len(df):,} JSON record(s)")
                filtered_df, date_filter = filter_loaded_data_by_selected_dates(df, "json", json_from, json_to)
                payload["data"]        = filtered_df
                payload["date_filter"] = date_filter
                st.dataframe(filtered_df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    return payload


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — DATA FILTERS
# ──────────────────────────────────────────────────────────────────────────────
def default_model_input() -> dict:
    return dict(
        zone="NCR", divcode="PRYJ", section="ALD-MGS",
        rail_section=52,   rail_section_bnn="52KG",
        track_type="BG",   straight_curve="Curved",
        service_life=8.0,  annual_gmt=35.0, gmt_carried=280.0,
        usfd_duration=280.0 / 35.0 * 365.25,
        speed=110,         axle_load=22,    axle_bnn="22.32",
        usfd_class="GR",   lrrr="Left",     line="DN",
        usfd_bnn="GR",     lrrr_bnn="Left", line_bnn="DN",
        track_bnn="BG",    straight_curve_bnn="Curved",
    )


def find_existing_column(df: pd.DataFrame, aliases: list[str]):
    canonical = {canonical_col(col): col for col in df.columns}
    for alias in aliases:
        col = canonical.get(canonical_col(alias))
        if col is not None:
            return col
    return None


def apply_sidebar_multiselect_filter(df: pd.DataFrame, label: str, aliases: list[str], key: str) -> pd.DataFrame:
    col = find_existing_column(df, aliases)
    if col is None or pd.api.types.is_numeric_dtype(df[col]):
        return df

    options      = sorted(str(v) for v in df[col].dropna().unique() if str(v).strip())[:300]
    numeric_frac = pd.to_numeric(pd.Series(options), errors="coerce").notna().mean() if options else 0
    if not options or numeric_frac > 0.8:
        return df

    selected = st.sidebar.multiselect(label, options, key=key)
    return df[df[col].astype(str).isin(selected)].copy() if selected else df


def apply_sidebar_zone_filter(df: pd.DataFrame) -> pd.DataFrame:
    col = find_existing_column(df, ZONE_COLUMN_ALIASES)
    if col is None:
        st.sidebar.caption("Zone filter unavailable: no zone column found.")
        return df

    data_zones = sorted(
        {zone_display_name(v) for v in df[col].dropna().unique() if str(v).strip()},
        key=lambda z: RAILWAY_ZONES.index(z) if z in RAILWAY_ZONES else len(RAILWAY_ZONES),
    )
    if not data_zones:
        return df

    selected = st.sidebar.multiselect("Railway Zone", data_zones, key="filter_zone")
    if not selected:
        return df

    selected_codes = {normalize_zone_code(z) for z in selected}
    return df.loc[df[col].apply(lambda v: normalize_zone_code(v) in selected_codes)].copy()


def render_data_filters(source_payload: dict) -> dict:
    df = source_payload.get("data")
    if df is None or df.empty:
        return default_model_input()

    st.sidebar.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.9rem;font-weight:800;
    letter-spacing:1.2px;text-transform:uppercase;color:#0f172a;
    padding:6px 0 14px;border-bottom:1px solid #d8dee8;margin-bottom:14px;'>
    Data Filters</div>""", unsafe_allow_html=True)

    original_count   = len(df)
    filtered_metric  = st.sidebar.empty()

    with st.sidebar.expander("Filter Loaded Data", expanded=True):
        filtered = df.copy()
        filtered = apply_sidebar_zone_filter(filtered)
        filtered = apply_sidebar_multiselect_filter(filtered, "Division",   ["DIVCODE","DIVISION","DIV"],                            "filter_division")
        filtered = apply_sidebar_multiselect_filter(filtered, "Section",    ["SECTION","SEC","ROUTE_SECTION"],                        "filter_section")
        filtered = apply_sidebar_multiselect_filter(filtered, "Track Type", ["TRACK_TYPE","TRACK"],                                   "filter_track")
        filtered = apply_sidebar_multiselect_filter(filtered, "Alignment",  ["STRAIGHT_CURVE","ALIGNMENT"],                          "filter_alignment")
        filtered = apply_sidebar_multiselect_filter(filtered, "USFD",       ["USFD_CLASSIFICATION","USFDCLASSIFICATION","USFD"],      "filter_usfd")
        filtered = apply_sidebar_multiselect_filter(filtered, "LRRR",       ["LRRR"],                                                 "filter_lrrr")
        filtered = apply_sidebar_multiselect_filter(filtered, "Line",       ["LINE"],                                                  "filter_line")

    source_payload["unfiltered_data"] = df
    source_payload["data"]            = filtered
    filtered_metric.metric("Filtered Records", f"{len(filtered):,}", f"of {original_count:,}")

    if filtered.empty:
        return default_model_input()

    inp = api_row_to_rf_input(filtered.iloc[0], default_model_input())
    inp["usfd_bnn"]          = inp.get("usfd_bnn") or inp["usfd_class"]
    inp["lrrr_bnn"]          = inp["lrrr"]
    inp["line_bnn"]          = inp.get("line_bnn", "DN")
    inp["track_bnn"]         = inp["track_type"]
    inp["straight_curve_bnn"] = inp["straight_curve"]
    return inp


def render_sidebar(source_payload: dict | None = None) -> dict:
    df = (source_payload or {}).get("data")
    if df is not None and not df.empty:
        return render_data_filters(source_payload)

    st.sidebar.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.9rem;font-weight:800;
    letter-spacing:1.2px;text-transform:uppercase;color:#0f172a;
    padding:6px 0 14px;border-bottom:1px solid #d8dee8;margin-bottom:14px;'>
    Data Filters</div>""", unsafe_allow_html=True)
    st.sidebar.info("Load data from Live API, Excel/CSV, or JSON to enable filters.")
    return default_model_input()


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 1 — RANDOM FOREST TAB
# ──────────────────────────────────────────────────────────────────────────────
def render_rf(inp: dict, source_payload: dict | None = None):
    st.markdown("""
    <div class='mod-banner'>
      <div class='mod-title rf'>Module 01 — Random Forest · Remaining Life Predictor</div>
      <div class='mod-sub'>All-India consolidated · RandomForestRegressor (100 trees) · Target: DURATION (days)</div>
    </div>""", unsafe_allow_html=True)

    rf_model, preprocessor, section_freq = load_rf_artifacts()
    model_ok = rf_model is not None and preprocessor is not None

    if not model_ok:
        st.markdown("""<div class='info-bar'>⚡ <b>Demo mode</b> — copy
        <code>random_forest_model.pkl</code>, <code>preprocessor.pkl</code>, and
        <code>section_freq.pkl</code> into <code>./models/</code> to run the trained model.</div>""",
        unsafe_allow_html=True)

    # Batch mode: data loaded
    df = source_payload.get("data") if source_payload else None
    if df is not None and not df.empty:
        total_records = len(df)
        step = 1 if total_records <= 500 else (25 if total_records <= 5000 else 100)
        st.markdown("<div class='info-bar'>Remaining life will be predicted for the filtered loaded data.</div>", unsafe_allow_html=True)

        c_show, c_run = st.columns([1, 3])
        with c_show:
            max_show = st.slider("Records to predict/show", 1, total_records, min(200, total_records), step=step, key="rf_records_to_show")
        with c_run:
            run_filtered = st.button("PREDICT FILTERED DATA", key="rf_run_filtered")

        if run_filtered:
            with st.spinner("Predicting remaining life..."):
                results, errors = predict_remaining_life_rows(
                    df, inp,
                    (source_payload.get("date_filter") or {}).get("date_col", ""),
                    int(max_show),
                )
                st.session_state["rf_filtered_result"] = results
                st.session_state["rf_filtered_errors"] = errors

        results = st.session_state.get("rf_filtered_result")
        errors  = st.session_state.get("rf_filtered_errors", [])
        if results is not None:
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Predicted Records", f"{len(results):,}")
            with c2: st.metric("Shortest Life", f"{int(results['Remaining Life Days'].min()):,} days" if not results.empty else "-")
            with c3: st.metric("Rows Skipped",   f"{len(errors):,}")
            st.dataframe(results, use_container_width=True, hide_index=True)
            if errors:
                with st.expander("Rows skipped during RF prediction"):
                    st.write("\n".join(errors[:200]))
        return

    # Single-record mode
    col_btn, _ = st.columns([1, 5])
    with col_btn:
        run = st.button("▶ PREDICT", key="rf_run")

    if run:
        with st.spinner("Running Random Forest prediction…"):
            try:
                st.session_state["rf_result"] = predict_rf(inp)
            except Exception as e:
                st.error(f"Prediction error: {e}")

    if "rf_result" in st.session_state:
        res  = st.session_state["rf_result"]
        days = res["remaining_days"]
        yrs  = res["remaining_years"]
        urg  = ("🟢 ADEQUATE" if yrs > 3 else "🟡 REVIEW SOON" if yrs > 1 else "🔴 URGENT RENEWAL")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='rbox'>
              <div class='rlabel'>Remaining Life</div>
              <div class='rvalue amber'>{days:,}</div>
              <div class='rsub'>DAYS</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='rbox blue'>
              <div class='rlabel'>Equivalent</div>
              <div class='rvalue blue'>{yrs}</div>
              <div class='rsub'>YEARS</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='rbox'>
              <div class='rlabel'>Urgency Status</div>
              <div style='font-size:1rem;font-weight:700;color:#111827;margin-top:10px;'>{urg}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(plot_gauge(days), use_container_width=True)

        with st.expander("📋 Input Summary & Engineered Features"):
            axle_enc = AXLE_MAP_RF.get(inp["axle_load"], 2)
            st.dataframe(pd.DataFrame({
                "Parameter": ["Zone", "Division", "Rail Section", "Axle Load", "Track Type", "Alignment",
                              "Service Life", "Annual GMT", "GMT Carried", "Speed",
                              "USFD", "LRRR", "Line", "→ ANNUAL_LOAD", "→ SPEED_AGMT", "→ SECTION_FREQ"],
                "Value": [
                    inp["zone"], inp.get("divcode","PRYJ"),
                    f"{inp['rail_section']} kg/m → {RAIL_SECTION_RF_MAP.get(inp['rail_section'])}",
                    f"{inp['axle_load']} t → {axle_enc}",
                    inp["track_type"], inp["straight_curve"],
                    f"{inp['service_life']} yr", inp["annual_gmt"], inp["gmt_carried"], f"{inp['speed']} km/h",
                    inp["usfd_class"], inp["lrrr"], inp["line"],
                    round(inp["annual_gmt"] * axle_enc, 2),
                    round(inp["speed"] * inp["annual_gmt"], 2),
                    round(section_freq.get(inp["section"], 0.0), 6),
                ],
            }), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 2 — BNN-COX TAB
# ──────────────────────────────────────────────────────────────────────────────
def render_bnn(inp: dict, source_payload: dict | None = None):
    st.markdown("""
    <div class='mod-banner'>
      <div class='mod-title bnn'>Module 02 — BNN-Cox · Survival &amp; Risk Analyser</div>
      <div class='mod-sub'>Zone-specific BNN-Cox artifacts · DenseReparameterization 64→32→1 + CoxPHFitter ·
      S(t) · Risk Score · Risk Category</div>
    </div>""", unsafe_allow_html=True)

    # Batch mode: data loaded
    df = source_payload.get("data") if source_payload else None
    if df is not None and not df.empty:
        total_records = len(df)
        zone_col = find_existing_column(df, ZONE_COLUMN_ALIASES)
        zones_in_data = (
            sorted({zone_display_name(v) for v in df[zone_col].dropna().unique() if str(v).strip()})
            if zone_col else [zone_display_name(inp.get("zone", "NCR"))]
        )
        missing_zones = [z for z in zones_in_data if not zone_has_cox_artifact(z)]
        if missing_zones:
            st.markdown(
                f"<div class='warn-bar'>Missing Cox PKL for: <b>{', '.join(missing_zones)}</b>. "
                "Those rows will use demo estimates.</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='info-bar'>Survival analysis will run on the filtered data "
            f"(<b>{total_records:,}</b> records).</div>", unsafe_allow_html=True)
        _render_risk_threshold_reference()

        c_rows, c_run = st.columns([2, 1])
        with c_rows:
            survival_rows = st.slider("Records to analyse", 1, total_records, total_records, key="bnn_records_to_show")
        with c_run:
            st.markdown("<br>", unsafe_allow_html=True)
            run_filtered_bnn = st.button("ANALYSE FILTERED DATA", key="bnn_run_filtered", use_container_width=True)

        if run_filtered_bnn:
            with st.spinner("Running survival analysis..."):
                results, all_curves, errors = predict_survival_rows(
                    df, inp, int(survival_rows),
                    (source_payload.get("date_filter") or {}).get("date_col", ""),
                )
                st.session_state["bnn_filtered_result"] = results
                st.session_state["bnn_filtered_curves"] = all_curves
                st.session_state["bnn_filtered_errors"] = errors

        results    = st.session_state.get("bnn_filtered_result")
        all_curves = st.session_state.get("bnn_filtered_curves", [])
        errors     = st.session_state.get("bnn_filtered_errors", [])
        if results is not None:
            st.markdown("---")
            risk_col = "5-Year Risk Score" if "5-Year Risk Score" in results.columns else "Risk Score"
            s1, s2, s3 = st.columns(3)
            with s1: st.metric("Analysed Records", f"{len(results):,}")
            with s2: st.metric("Highest Risk",     f"{results[risk_col].max():.3f}" if not results.empty else "-")
            with s3: st.metric("Rows Skipped",     f"{len(errors):,}")
            st.dataframe(results, use_container_width=True, hide_index=True)
            if all_curves:
                st.pyplot(plot_survival_all(all_curves, SURVIVAL_PLOT_POINTS), use_container_width=True)
            if errors:
                with st.expander("Rows skipped during survival analysis"):
                    st.write("\n".join(errors[:200]))
        return

    # Single-record mode
    bnn_model, cph, scaler = load_bnn_artifacts(inp.get("zone", "NCR"))
    model_ok = cph is not None
    expected = expected_bnn_filenames(inp.get("zone", "NCR"))

    if not model_ok:
        st.markdown(f"""<div class='info-bar'>Demo mode — add
        <code>{expected['Cox']}</code>, <code>{expected['BNN']}</code>, and
        <code>{expected['Scaler']}</code> to run the trained model.</div>""", unsafe_allow_html=True)
    elif bnn_model is None:
        st.markdown(
            f"<div class='info-bar'>ℹ️ BNN model not found (<code>{expected['BNN']}</code>). "
            "Running in <b>Cox-only mode</b> with neutral g_bnn.</div>", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        run = st.button("▶ SURVIVAL ANALYSIS", key="bnn_run")

    if run:
        with st.spinner("Running BNN-Cox survival analysis…"):
            if model_ok:
                result, err = predict_bnn_result(inp)
                if err:
                    st.error(f"Survival analysis error: {err}")
                else:
                    st.session_state["bnn_result"] = result
            else:
                st.session_state["bnn_result"] = demo_bnn(inp)

    if "bnn_result" in st.session_state:
        res = st.session_state["bnn_result"]
        rs  = res["risk_score"]
        cat = res["risk_category"]

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""<div class='rbox blue'>
              <div class='rlabel'>Risk Score (Relative)</div>
              <div class='rvalue blue'>{rs:.3f}</div>
              <div class='rsub'>Calibrated partial hazard [0–1]</div></div>""", unsafe_allow_html=True)

        t_arr = np.array(res["time"]); s_arr = np.array(res["survival"])
        cross = np.where(s_arr <= 0.5)[0]
        med   = f"{t_arr[cross[0]] / 365:.1f} yr" if len(cross) else "> 15 yr"
        with c2:
            st.markdown(f"""<div class='rbox purple'>
              <div class='rlabel'>Median Survival</div>
              <div class='rvalue purple' style='font-size:1.6rem;'>{med}</div>
              <div class='rsub'>S(t) = 0.50 crossover</div></div>""", unsafe_allow_html=True)

        sv1y   = res["horizons"].get(365, 0)
        col_1y = "green" if sv1y > 0.65 else ("amber" if sv1y > 0.35 else "red")
        with c3:
            st.markdown(f"""<div class='rbox {col_1y}'>
              <div class='rlabel'>1-Year Survival</div>
              <div class='rvalue {col_1y}'>{sv1y:.1%}</div>
              <div class='rsub'>P(no failure ≤ 1 yr)</div></div>""", unsafe_allow_html=True)

        g_disp    = f"{res['g_bnn']:.4f}" if res.get("g_bnn") is not None else "N/A"
        mode_disp = res.get("model_mode", "")
        with c4:
            st.markdown(f"""<div class='rbox'>
              <div class='rlabel'>BNN g(x) Output</div>
              <div class='rvalue amber' style='font-size:1.5rem;'>{g_disp}</div>
              <div class='rsub'>Mode: {mode_disp}</div></div>""", unsafe_allow_html=True)

        st.markdown(
            f"<div style='margin:14px 0 8px;'>"
            f"<span style='color:#475569;font-size:0.84rem;font-family:Space Mono,monospace;'>RISK CATEGORY &nbsp;</span>"
            f"<span class='rbadge r-{cat.lower()}'>{cat}</span></div>",
            unsafe_allow_html=True)

        _render_risk_threshold_reference()
        st.pyplot(plot_survival(res, SURVIVAL_PLOT_POINTS), use_container_width=True)

        st.markdown("<div class='sec-hdr' style='margin-top:20px;'>Survival Probabilities at Key Horizons</div>", unsafe_allow_html=True)
        h_lbls = ["6 Months", "1 Year", "2 Years", "3 Years", "5 Years", "10 Years"]
        st.dataframe(pd.DataFrame([
            {"Horizon": lbl, "Days": h_d, "Survival Prob.": f"{sv:.2%}",
             "Status": "✅ Safe" if sv > 0.65 else ("⚠️ Monitor" if sv > 0.35 else "❌ At Risk")}
            for (h_d, sv), lbl in zip(res["horizons"].items(), h_lbls)
        ]), use_container_width=True, hide_index=True)

        with st.expander("📋 BNN-Cox Input Detail"):
            axle_ord = AXLE_ORDER_BNN.index(inp["axle_bnn"])
            rail_ord = RAIL_ORDER_BNN.index(inp["rail_section_bnn"])
            st.dataframe(pd.DataFrame({
                "Parameter": ["Division", "Section",
                              "RAIL_SECTION (BNN ord.)", "AXLELOAD (BNN ord.)",
                              "Annual GMT", "GMT Carried", "Speed (km/h)", "Service Life (yr)",
                              "USFD (BNN)", "LRRR (BNN)", "Line (BNN)", "Track Type", "Alignment",
                              "TRACK_TYPE_LWR", "TRACK_TYPE_SWR", "STRAIGHT_CURVE_STRAIGHT",
                              "BNN g(x)", "Cox Partial Hazard", "Model Mode"],
                "Value": [
                    inp["divcode"], inp["section"],
                    f"'{inp['rail_section_bnn']}' → {rail_ord}",
                    f"'{inp['axle_bnn']}' → {axle_ord}",
                    inp["annual_gmt"], inp["gmt_carried"], inp["speed"], inp["service_life"],
                    inp["usfd_bnn"], inp["lrrr_bnn"], inp["line_bnn"],
                    inp["track_bnn"], inp["straight_curve_bnn"],
                    1 if inp["track_bnn"] == "LWR" else 0,
                    1 if inp["track_bnn"] == "SWR" else 0,
                    1 if inp["straight_curve_bnn"] == "Straight" else 0,
                    res.get("g_bnn", "N/A"), res.get("partial_hazard", "N/A"), res.get("model_mode", "N/A"),
                ],
            }), use_container_width=True, hide_index=True)


def _render_risk_threshold_reference():
    st.markdown("""
    <div class='mod-banner' style='padding:14px 18px;margin:10px 0 16px;'>
      <div class='mod-title bnn' style='font-size:0.9rem;'>Relative Risk Score Thresholds
        <span style='font-family:DM Sans,sans-serif;font-size:0.78rem;font-weight:600;color:#334155;'>
          — derived from calibrated Cox partial hazard sigmoid
        </span>
      </div>
      <div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:12px;'>
        <div class='rbox green' style='margin:0;padding:12px 14px;'>
          <div class='rlabel'>Low Risk</div>
          <div style='font-size:1.05rem;font-weight:800;color:#166534;'>0.00 – 0.349</div>
          <div style='font-size:0.8rem;color:#166534;margin-top:4px;'>partial hazard &lt; 1.0</div>
        </div>
        <div class='rbox' style='margin:0;padding:12px 14px;border-left-color:#d97706;'>
          <div class='rlabel'>Medium Risk</div>
          <div style='font-size:1.05rem;font-weight:800;color:#92400e;'>0.350 – 0.649</div>
          <div style='font-size:0.8rem;color:#92400e;margin-top:4px;'>partial hazard ≈ 1.0 – 3.0</div>
        </div>
        <div class='rbox red' style='margin:0;padding:12px 14px;'>
          <div class='rlabel'>High Risk</div>
          <div style='font-size:1.05rem;font-weight:800;color:#991b1b;'>0.650 – 1.000</div>
          <div style='font-size:0.8rem;color:#991b1b;margin-top:4px;'>partial hazard &gt; 3.0</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA ALERTS TAB
# ──────────────────────────────────────────────────────────────────────────────
def render_api_alerts(source_payload: dict, inp: dict):
    source_name    = source_payload.get("source", "Selected source")
    threshold_days = int(st.session_state.get("remaining_life_alert_threshold", URGENT_REMAINING_LIFE_DAYS))

    st.markdown(f"""
    <div class='mod-banner'>
      <div class='mod-title rf'>Data Batch Screening — Remaining Life ≤ {threshold_days} Days</div>
      <div class='mod-sub'>Run RF remaining-life screening on the loaded data source.</div>
    </div>""", unsafe_allow_html=True)

    df          = source_payload.get("data")
    date_filter = source_payload.get("date_filter") or {}

    if df is None or df.empty:
        if date_filter.get("enabled") and date_filter.get("original_count", 0) > 0:
            msg = "No records match the selected date range. Adjust the filter in the Data Source panel."
        elif source_name == "Live API (IRCEP)":
            msg = "No API records loaded yet. Use <b>Fetch</b> in the Data Source panel."
        elif source_name == "Upload Excel/CSV":
            msg = "No file loaded yet. Upload a file in the Data Source panel."
        else:
            msg = "No JSON records loaded yet. Paste valid JSON in the Data Source panel."
        st.markdown(f"<div class='warn-bar'>{msg}</div>", unsafe_allow_html=True)
        return

    data_signature = (
        source_name, len(df),
        tuple(str(c) for c in df.columns),
        date_filter.get("enabled"),
        str(date_filter.get("date_col", "")),
        str(date_filter.get("start", "")),
        str(date_filter.get("end", "")),
        threshold_days,
    )
    if st.session_state.get("api_remaining_life_signature") != data_signature:
        st.session_state.pop("api_remaining_life_alerts", None)
        st.session_state.pop("api_remaining_life_errors", None)

    rf_model, preprocessor, _ = load_rf_artifacts()
    if rf_model is None or preprocessor is None:
        st.markdown("<div class='info-bar'>Demo mode active — RF artifacts missing.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        st.metric("Records in Range", f"{len(df):,}")
    with c2:
        threshold_days = st.number_input("Alert Limit (days)", min_value=1, max_value=3650,
                                         value=threshold_days, step=5, key="remaining_life_alert_threshold")
    with c3:
        run_batch = st.button("ANALYSE LOADED DATA", key="analyse_api_records")

    if date_filter.get("enabled"):
        date_col_lbl = date_filter.get("date_col") or "API-fetched date range"
        st.caption(f"Analysing records from {date_filter.get('start')} to {date_filter.get('end')} | {date_col_lbl}")

    if run_batch:
        with st.spinner("Predicting remaining life for all loaded records..."):
            alerts, errors = predict_api_remaining_life(df, inp, threshold_days, date_filter.get("date_col", ""))
            st.session_state["api_remaining_life_alerts"]    = alerts
            st.session_state["api_remaining_life_errors"]    = errors
            st.session_state["api_remaining_life_signature"] = data_signature

    alerts = st.session_state.get("api_remaining_life_alerts")
    errors = st.session_state.get("api_remaining_life_errors", [])
    if alerts is None:
        st.markdown("<div class='info-bar'>Click <b>ANALYSE LOADED DATA</b> to screen records.</div>", unsafe_allow_html=True)
        return

    st.markdown("---")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f"""<div class='rbox red'>
          <div class='rlabel'>Rails ≤ {threshold_days} Days</div>
          <div class='rvalue red'>{len(alerts):,}</div>
          <div class='rsub'>Requires urgent review</div></div>""", unsafe_allow_html=True)
    with a2:
        st.markdown(f"""<div class='rbox blue'>
          <div class='rlabel'>Records Analysed</div>
          <div class='rvalue blue'>{len(df):,}</div>
          <div class='rsub'>{source_name}</div></div>""", unsafe_allow_html=True)
    with a3:
        st.markdown(f"""<div class='rbox'>
          <div class='rlabel'>Rows Skipped</div>
          <div class='rvalue amber'>{len(errors):,}</div>
          <div class='rsub'>Could not be mapped/predicted</div></div>""", unsafe_allow_html=True)

    if alerts.empty:
        st.success("No records within the alert threshold.")
    else:
        st.dataframe(alerts, use_container_width=True, hide_index=True)
        st.download_button(
            "DOWNLOAD ALERTS CSV",
            data=alerts.to_csv(index=False).encode("utf-8"),
            file_name="remaining_life_alerts.csv",
            mime="text/csv",
            key="download_api_alerts",
        )
    if errors:
        with st.expander("Rows skipped during analysis"):
            st.write("\n".join(errors[:200]))


# ──────────────────────────────────────────────────────────────────────────────
# MODEL INFO TAB
# ──────────────────────────────────────────────────────────────────────────────
def render_info():
    st.markdown("<div class='sec-hdr'>Model Architectures</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='mod-banner'>
          <div class='mod-title rf'>Random Forest Model</div>
          <div style='color:#334155;font-size:0.9rem;line-height:1.75;margin-top:10px;'>
            <b style='color:#111827;'>Scope</b> · All-India (all zones)<br>
            <b style='color:#111827;'>Filter</b> · Weather_failed == 1, DURATION IQR clip<br>
            <b style='color:#111827;'>Estimators</b> · 100 trees, random_state=42<br>
            <b style='color:#111827;'>Preprocessing</b><br>
            &nbsp; StandardScaler → RAIL_SECTION, SPEED, GMTCARRIED, AXLELOAD, SERVICE_LIFE, ANNUAL_LOAD, SPEED_AGMT<br>
            &nbsp; BinaryEncoder → SECTION (high-cardinality)<br>
            &nbsp; OrdinalEncoder → USFD_CLASSIFICATION [GR &lt; OBS JP &lt; OBS &lt; IMR]<br>
            &nbsp; OHE drop=first → LINE, LRRR, TRACK_TYPE, STRAIGHT_CURVE<br>
            <b style='color:#111827;'>Engineered Features</b><br>
            &nbsp; ANNUAL_LOAD = ANNUAL_GMT × AXLELOAD<br>
            &nbsp; SPEED_AGMT = SPEED × ANNUAL_GMT
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='mod-banner'>
          <div class='mod-title bnn'>BNN-Cox Model</div>
          <div style='color:#334155;font-size:0.9rem;line-height:1.75;margin-top:10px;'>
            <b style='color:#111827;'>Scope</b> · NCR zone (AGRA / JHS / PRYJ)<br>
            <b style='color:#111827;'>BNN Architecture</b> · DenseReparameterization 64→32→1<br>
            &nbsp; Prior: N(0,1) · KL weight: 1e-5 · Optimizer: RMSprop(0.01)<br>
            &nbsp; Loss: Negative Log Partial Likelihood · EarlyStopping(patience=10)<br>
            <b style='color:#111827;'>Cox Formula</b><br>
            &nbsp; GMT_CARRIED + RAIL_SECTION + TRACK_TYPE_LWR + TRACK_TYPE_SWR<br>
            &nbsp; + STRAIGHT_CURVE_STRAIGHT + g_bnn<br>
            <b style='color:#111827;'>Preprocessing</b><br>
            &nbsp; OrdinalEncoder → RAIL_SECTION, AXLELOAD<br>
            &nbsp; OHE drop=first → DIVCODE, LINE, USFD, LRRR, TRACK_TYPE, STRAIGHT_CURVE<br>
            &nbsp; StandardScaler → full feature vector before BNN
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-hdr' style='margin-top:18px;'>Artifact Placement</div>", unsafe_allow_html=True)
    st.code("""
models/
├── random_forest_model.pkl   # RandomForestRegressor (RF notebook)
├── preprocessor.pkl          # ColumnTransformer x_mapper (RF notebook)
├── section_freq.pkl          # dict{section: norm_freq} from train split
├── bnn_model_NCR.h5          # tf_keras BNN H5 model (BNN notebook) [optional]
├── cph_NCR.pkl               # lifelines CoxPHFitter (BNN notebook)
└── scaler_NCR.pkl            # StandardScaler for BNN input [optional]
    """, language="text")

    st.markdown("<div class='sec-hdr' style='margin-top:18px;'>Complete Feature Reference</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"Input Field": "Rail Section",        "RF Treatment": "strip 'KG' → int → map {52→1, 60→2} → StandardScaler",    "BNN Treatment": "OrdinalEncoder ['52KG'→0, '60KG'→1]"},
        {"Input Field": "Axle Load",           "RF Treatment": "map {20→1, 22→2, 23→3, 25→4} → StandardScaler",           "BNN Treatment": "OrdinalEncoder ['Up to 20.32'→0 … '25'→3]"},
        {"Input Field": "Speed",               "RF Treatment": "StandardScaler",                                            "BNN Treatment": "passthrough num → StandardScaler"},
        {"Input Field": "GMT Carried",         "RF Treatment": "StandardScaler",                                            "BNN Treatment": "passthrough num → StandardScaler"},
        {"Input Field": "Annual GMT",          "RF Treatment": "used to compute ANNUAL_LOAD & SPEED_AGMT",                 "BNN Treatment": "passthrough num → StandardScaler"},
        {"Input Field": "Service Life",        "RF Treatment": "StandardScaler",                                            "BNN Treatment": "passthrough num → StandardScaler"},
        {"Input Field": "USFD Classification", "RF Treatment": "OrdinalEncoder [GR < OBS JP < OBS < IMR]",                 "BNN Treatment": "OHE drop=first (DFW dropped)"},
        {"Input Field": "LRRR",                "RF Treatment": "Left / Right categorical",                                  "BNN Treatment": "Left / Right categorical"},
        {"Input Field": "Track Type",          "RF Treatment": "OHE drop=first (BG dropped → LWR, SWR)",                   "BNN Treatment": "OHE drop=first → used in Cox formula"},
        {"Input Field": "Alignment",           "RF Treatment": "OHE drop=first (Curved dropped → Straight)",               "BNN Treatment": "OHE drop=first → Cox formula"},
        {"Input Field": "Line",                "RF Treatment": "reduce_line(): DN/UP/SL/OT → OHE drop=first",              "BNN Treatment": "DN/OTH/SL/UP → OHE drop=first (DN dropped)"},
        {"Input Field": "Section",             "RF Treatment": "BinaryEncoder (high-cardinality)",                          "BNN Treatment": "SECTION_FREQ = value_counts() appended outside pipeline"},
        {"Input Field": "Division (DIVCODE)",  "RF Treatment": "dropped (not used)",                                        "BNN Treatment": "OHE drop=first (AGRA dropped → JHS, PRYJ)"},
        {"Input Field": "ANNUAL_LOAD ★",       "RF Treatment": "ANNUAL_GMT × AXLELOAD → StandardScaler",                  "BNN Treatment": "—"},
        {"Input Field": "SPEED_AGMT ★",        "RF Treatment": "SPEED × ANNUAL_GMT → StandardScaler",                     "BNN Treatment": "—"},
        {"Input Field": "g_bnn ★",             "RF Treatment": "—",                                                         "BNN Treatment": "BNN non-linear output → Cox covariate (training-mean neutral value if BNN absent)"},
        {"Input Field": "Risk Score ★",        "RF Treatment": "—",                                                         "BNN Treatment": "sigmoid(2.5·log(partial_hazard)) — calibrated relative risk [0,1]; replaces raw 1-S(5yr)"},
    ]), use_container_width=True, hide_index=True)
    st.caption("★ = engineered / derived feature")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class='app-header'>
      <h1>🛤️ Rail Failure Analysis System</h1>
      <p>Random Forest remaining life predictor (all-India) &nbsp;+&nbsp;
         BNN-Cox survival &amp; risk analyser (NCR zone) &nbsp;|&nbsp;
         Indian Railways track infrastructure</p>
    </div>""", unsafe_allow_html=True)

    source_payload = render_data_source()
    with st.expander("PKL files read by this app", expanded=True):
        st.dataframe(pd.DataFrame(artifact_status_rows()), use_container_width=True, hide_index=True)

    inp = render_sidebar(source_payload)
    inp["data_source"] = source_payload["source"]

    tab1, tab2, tab3, tab4 = st.tabs(["RF Remaining Life", "BNN-Cox Survival", "Data Alerts", "Model Information"])
    with tab1: render_rf(inp, source_payload)
    with tab2: render_bnn(inp, source_payload)
    with tab3: render_api_alerts(source_payload, inp)
    with tab4: render_info()


if __name__ == "__main__":
    main()
