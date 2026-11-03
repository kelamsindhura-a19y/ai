"""
Configuration Module for Prompt Engineering Comparison System
CSE 2nd Year Mini-Capstone Project (Course: 25CS301)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Detect serverless / Vercel cloud environment
IS_VERCEL = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

# On Vercel, the root filesystem is read-only, only /tmp is writable
if IS_VERCEL:
    RESULTS_DIR = Path("/tmp/results")
else:
    RESULTS_DIR = BASE_DIR / "results"

# Safely ensure directories exist without crashing on read-only systems
try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# File paths
QUESTIONS_FILE = DATASET_DIR / "questions.csv"
EXPERIMENTS_FILE = RESULTS_DIR / "experiments.csv"

# Categories
CATEGORIES = [
    "General Knowledge",
    "Mathematics",
    "Logical Reasoning",
    "Science",
    "Programming",
]

# Supported Gemini Models
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Prompting Techniques
TECHNIQUES = [
    "Zero-Shot",
    "Few-Shot",
    "Chain-of-Thought",
]

# Evaluation Weights (Total = 1.0)
WEIGHTS = {
    "correctness": 0.50,
    "relevance": 0.25,
    "explanation": 0.25,
}

# UI Styling Theme Colors (Modern Blue & Purple AI Theme)
THEME_COLORS = {
    "primary": "#6366F1",       # Indigo / Purple
    "secondary": "#3B82F6",     # Vibrant Blue
    "accent": "#8B5CF6",        # Violet
    "success": "#10B981",       # Emerald Green
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Crimson Red
    "background_dark": "#0F172A",
    "card_dark": "#1E293B",
    "text_light": "#F8FAFC",
}

# Technique color mapping for charts
TECHNIQUE_COLORS = {
    "Zero-Shot": "#3B82F6",         # Sky Blue
    "Few-Shot": "#8B5CF6",          # Purple
    "Chain-of-Thought": "#EC4899",  # Pink/Rose
}
