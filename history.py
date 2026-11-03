"""
History Module for Prompt Engineering Comparison
Manages CSV persistence, analytics summarization, and export of experimental runs.
Includes in-memory cache fallback for serverless read-only environments (Vercel).
"""

import os
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional
from config import EXPERIMENTS_FILE

# In-memory session store fallback for serverless runtimes
_MEMORY_HISTORY = []

# Standard schema for experimental results
CSV_COLUMNS = [
    "experiment_id",
    "timestamp",
    "category",
    "question",
    "reference_answer",
    "technique",
    "response_text",
    "correctness",
    "status",
    "relevance",
    "explanation",
    "overall_score",
    "latency",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "is_demo",
    "model_name"
]


def initialize_history_file() -> None:
    """Safely ensures results directory and experiments.csv file exist."""
    try:
        EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not EXPERIMENTS_FILE.exists():
            df = pd.DataFrame(columns=CSV_COLUMNS)
            df.to_csv(EXPERIMENTS_FILE, index=False)
    except Exception:
        pass


def save_experiment_results(
    experiment_id: str,
    question: str,
    category: str,
    reference_answer: str,
    evaluations: Dict[str, Dict[str, Any]],
    raw_responses: Dict[str, Dict[str, Any]],
    model_name: str = "gemini-2.5-flash"
) -> bool:
    """
    Saves an experimental trial with all three techniques to the CSV ledger
    and in-memory serverless cache.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for technique, ev in evaluations.items():
        resp_obj = raw_responses.get(technique, {})
        row = {
            "experiment_id": experiment_id,
            "timestamp": now_str,
            "category": category,
            "question": question,
            "reference_answer": reference_answer,
            "technique": technique,
            "response_text": resp_obj.get("response", ""),
            "correctness": ev.get("correctness", 0.0),
            "status": ev.get("status", "Unknown"),
            "relevance": ev.get("relevance", 0.0),
            "explanation": ev.get("explanation", 0.0),
            "overall_score": ev.get("overall_score", 0.0),
            "latency": ev.get("latency", 0.0),
            "prompt_tokens": ev.get("prompt_tokens", 0),
            "completion_tokens": ev.get("completion_tokens", 0),
            "total_tokens": ev.get("total_tokens", 0),
            "is_demo": ev.get("is_demo", False),
            "model_name": model_name
        }
        rows.append(row)
        _MEMORY_HISTORY.append(row)

    try:
        initialize_history_file()
        new_df = pd.DataFrame(rows)
        new_df.to_csv(EXPERIMENTS_FILE, mode="a", header=not EXPERIMENTS_FILE.exists(), index=False)
    except Exception:
        pass
    return True


def load_history() -> pd.DataFrame:
    """Loads historical experiment records safely from CSV or in-memory fallback."""
    try:
        initialize_history_file()
        if EXPERIMENTS_FILE.exists():
            df = pd.read_csv(EXPERIMENTS_FILE)
            if not df.empty:
                numeric_cols = ["correctness", "relevance", "explanation", "overall_score", "latency", "prompt_tokens", "completion_tokens", "total_tokens"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                return df
    except Exception:
        pass

    if _MEMORY_HISTORY:
        return pd.DataFrame(_MEMORY_HISTORY)

    return pd.DataFrame(columns=CSV_COLUMNS)


def clear_history() -> bool:
    """Clears history from memory and file."""
    global _MEMORY_HISTORY
    _MEMORY_HISTORY = []
    try:
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(EXPERIMENTS_FILE, index=False)
        return True
    except Exception:
        return True


def get_history_summary() -> Dict[str, Any]:
    """Computes high-level aggregate statistics safely."""
    df = load_history()
    if df.empty:
        return {
            "total_trials": 0,
            "unique_questions": 0,
            "avg_accuracy_by_tech": {},
            "avg_latency_by_tech": {},
            "best_accuracy_tech": "Chain-of-Thought",
            "fastest_tech": "Zero-Shot"
        }

    total_trials = len(df["experiment_id"].unique()) if "experiment_id" in df.columns else len(df) // 3
    unique_questions = len(df["question"].unique()) if "question" in df.columns else 0

    avg_acc = df.groupby("technique")["overall_score"].mean().to_dict() if "overall_score" in df.columns else {}
    avg_lat = df.groupby("technique")["latency"].mean().to_dict() if "latency" in df.columns else {}

    best_tech = max(avg_acc, key=avg_acc.get) if avg_acc else "Chain-of-Thought"
    fastest_tech = min(avg_lat, key=avg_lat.get) if avg_lat else "Zero-Shot"

    return {
        "total_trials": total_trials,
        "unique_questions": unique_questions,
        "avg_accuracy_by_tech": {k: round(v, 1) for k, v in avg_acc.items()},
        "avg_latency_by_tech": {k: round(v, 3) for k, v in avg_lat.items()},
        "best_accuracy_tech": best_tech,
        "fastest_tech": fastest_tech
    }
