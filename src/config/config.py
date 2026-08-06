from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ==========================================================
# DATA DIRECTORIES
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ==========================================================
# MODEL DIRECTORIES
# ==========================================================

MODELS_DIR = PROJECT_ROOT / "models"

TRANSPORTATION_MODEL_DIR = MODELS_DIR / "transportation"
ETA_MODEL_DIR = MODELS_DIR / "eta"
MAINTENANCE_MODEL_DIR = MODELS_DIR / "maintenance"
YOLO_MODEL_DIR = MODELS_DIR / "yolo"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# ==========================================================
# DATABASE
# ==========================================================

DATABASE_DIR = PROJECT_ROOT / "database"

# ==========================================================
# LOGS
# ==========================================================

LOGS_DIR = PROJECT_ROOT / "logs"

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# APPLICATION
# ==========================================================

APP_NAME = "SmartLogix AI"

APP_VERSION = "1.0.0"

# ==========================================================
# API
# ==========================================================

API_HOST = "127.0.0.1"

API_PORT = 8000