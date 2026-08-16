"""
config.py
Центральное место для всех настроек проекта.
"""

import os

# --- Пути ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "attendance.db")
TEST_DATASET_DIR = os.path.join(BASE_DIR, "data", "test_dataset")
EVAL_RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")

# --- Параметры лица ---
FACE_SIZE = (160, 160)
EMBEDDING_SIZE = 512

# --- Параметры распознавания ---
SIMILARITY_THRESHOLD = 0.6

# --- Параметры регистрации ---
NUM_REGISTRATION_PHOTOS = 3
