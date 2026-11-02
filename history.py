"""
History Module for Prompt Engineering Comparison
Manages CSV persistence, analytics summarization, and export of experimental runs.
"""

import os
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional
from config import EXPERIMENTS_FILE

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
    """Ensures results directory and experiments.csv file exist with headers."""
    EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not EXPERIMENTS_FILE.exists():
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(EXPERIMENTS_FILE, index=False)


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
    Saves an experimental trial with all three techniques to the CSV ledger.
    """
    initialize_history_file()
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

    new_df = pd.DataFrame(rows)
    # Append to existing CSV
    new_df.to_csv(EXPERIMENTS_FILE, mode="a", header=not EXPERIMENTS_FILE.exists(), index=False)
    return True


def load_history() -> pd.DataFrame:
    """Loads all historical experiment records as a Pandas DataFrame."""
    initialize_history_file()
    try:
        df = pd.read_csv(EXPERIMENTS_FILE)
        if df.empty:
            return pd.DataFrame(columns=CSV_COLUMNS)
        # Ensure numerical types
        numeric_cols = ["correctness", "relevance", "explanation", "overall_score", "latency", "prompt_tokens", "completion_tokens", "total_tokens"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df
    except Exception as e:
        print(f"Error loading history: {e}")
        return pd.DataFrame(columns=CSV_COLUMNS)


def clear_history() -> bool:
    """Clears all records from experiments.csv and reinitializes with column headers."""
    try:
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(EXPERIMENTS_FILE, index=False)
        return True
    except Exception as e:
        print(f"Error clearing history: {e}")
        return False


def get_history_summary() -> Dict[str, Any]:
    """Computes high-level aggregate statistics across all logged experiments."""
    df = load_history()
    if df.empty:
        return {
            "total_trials": 0,
            "unique_questions": 0,
            "avg_accuracy_by_tech": {},
            "avg_latency_by_tech": {},
            "best_accuracy_tech": "N/A",
            "fastest_tech": "N/A"
        }

    total_trials = len(df["experiment_id"].unique()) if "experiment_id" in df.columns else len(df) // 3
    unique_questions = len(df["question"].unique()) if "question" in df.columns else 0

    avg_acc = df.groupby("technique")["overall_score"].mean().to_dict()
    avg_lat = df.groupby("technique")["latency"].mean().to_dict()

    best_tech = max(avg_acc, key=avg_acc.get) if avg_acc else "N/A"
    fastest_tech = min(avg_lat, key=avg_lat.get) if avg_lat else "N/A"

    return {
        "total_trials": total_trials,
        "unique_questions": unique_questions,
        "avg_accuracy_by_tech": {k: round(v, 1) for k, v in avg_acc.items()},
        "avg_latency_by_tech": {k: round(v, 3) for k, v in avg_lat.items()},
        "best_accuracy_tech": best_tech,
        "fastest_tech": fastest_tech
    }
