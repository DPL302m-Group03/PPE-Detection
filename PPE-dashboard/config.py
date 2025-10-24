from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "output"
OUT_FRAMES = OUT_DIR / "frames"
OUT_VIOLATIONS = OUT_DIR / "violations"
OUT_REPORTS = OUT_DIR / "reports"

# --- Create Directories ---
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FRAMES.mkdir(parents=True, exist_ok=True)
OUT_VIOLATIONS.mkdir(parents=True, exist_ok=True)
OUT_REPORTS.mkdir(parents=True, exist_ok=True)

# --- Label Map and Colors ---
# Consistent mapping for all YOLO models
LABEL_MAP = {
    0: "worker",
    1: "helmet",
    2: "vest",
    3: "gloves",
    4.0: "boots", # Handle potential float keys
    4: "boots",
    5: "no_helmet",
    6: "no_vest",
    7: "no_gloves",
    8: "no_boots",
}

COLORS = {
    'worker': (60, 180, 75),
    'helmet': (255, 225, 25),
    'vest': (0, 130, 200),
    'gloves': (245, 130, 48),
    'boots': (145, 30, 180),
    'no_helmet': (230, 25, 75),
    'no_vest': (128, 128, 128),
    'no_gloves': (70, 240, 240),
    'no_boots': (210, 245, 60),
}