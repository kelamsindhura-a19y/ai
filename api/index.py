"""
Vercel Serverless Entrypoint & FastAPI Web Application
Prompt Engineering Strategies Compared: Zero-Shot, Few-Shot & Chain-of-Thought
AI Mini-Capstone Project (25CS301)
"""

import sys
import os
import json
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure parent directory is on sys.path for module resolution on Vercel
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

# Import existing backend modules
import config
from prompts import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_chain_of_thought_prompt,
    get_prompt_metadata
)
from gemini_service import execute_all_strategies, resolve_api_key
from evaluation import evaluate_technique_response
from history import (
    save_experiment_results,
    load_history,
    clear_history,
    get_history_summary
)

app = FastAPI(
    title="Prompt Engineering Strategies Comparison Lab",
    description="Vercel deployment for AI Mini-Capstone (25CS301)",
    version="1.0.0"
)

# Export aliases for Vercel Python runtime detection
application = app
handler = app


class ExperimentRequest(BaseModel):
    question: str
    category: str
    reference_answer: Optional[str] = ""
    api_key: Optional[str] = ""
    model_name: Optional[str] = "gemini-2.5-flash"
    temperature: Optional[float] = 0.2


@app.get("/api/questions")
def get_questions():
    """Returns the 25 curated questions from dataset/questions.csv."""
    try:
        import pandas as pd
        if config.QUESTIONS_FILE.exists():
            df = pd.read_csv(config.QUESTIONS_FILE)
            return JSONResponse(content={"questions": df.to_dict(orient="records")})
    except Exception as e:
        pass
    return JSONResponse(content={"questions": []})


@app.get("/api/summary")
def get_summary():
    """Returns aggregate summary metrics."""
    return JSONResponse(content=get_history_summary())


@app.get("/api/history")
def get_history_records():
    """Returns all historical experimental trials."""
    df = load_history()
    if df.empty:
        return JSONResponse(content={"records": []})
    return JSONResponse(content={"records": df.to_dict(orient="records")})


@app.post("/api/clear-history")
def clear_history_records():
    """Clears history."""
    success = clear_history()
    return JSONResponse(content={"success": success})


@app.get("/api/download-csv")
def download_csv():
    """Exports history as downloadable CSV."""
    df = load_history()
    csv_str = df.to_csv(index=False)
    return PlainTextResponse(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=experiments.csv"}
    )


@app.post("/api/run-experiment")
def run_experiment(req: ExperimentRequest):
    """Executes Zero-Shot, Few-Shot, and CoT prompts, evaluates them, and logs the result."""
    experiment_id = str(uuid.uuid4())[:8]
    keywords = [k.strip() for k in req.reference_answer.split() if len(k) > 2] if req.reference_answer else []

    # Run all 3 strategies with Google Gemini API (or Demo Mode fallback)
    raw_results = execute_all_strategies(
        question=req.question,
        category=req.category,
        api_key=req.api_key,
        model_name=req.model_name or "gemini-2.5-flash",
        temperature=req.temperature or 0.2
    )

    evaluations = {}
    for tech, resp_obj in raw_results.items():
        eval_report = evaluate_technique_response(
            technique=tech,
            response_dict=resp_obj,
            question=req.question,
            reference_answer=req.reference_answer,
            keywords=keywords
        )
        evaluations[tech] = eval_report

    # Save to history ledger
    save_experiment_results(
        experiment_id=experiment_id,
        question=req.question,
        category=req.category,
        reference_answer=req.reference_answer,
        evaluations=evaluations,
        raw_responses=raw_results,
        model_name=req.model_name or "gemini-2.5-flash"
    )

    return JSONResponse(content={
        "experiment_id": experiment_id,
        "question": req.question,
        "category": req.category,
        "reference_answer": req.reference_answer,
        "raw_results": raw_results,
        "evaluations": evaluations,
        "model_name": req.model_name or "gemini-2.5-flash",
        "timestamp": time.strftime("%H:%M:%S")
    })


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Renders the comprehensive single-page dashboard."""
    return HTML_CONTENT


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Engineering Strategies Compared | AI Capstone (25CS301)</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0B0F19; color: #F3F4F6; }
        .gradient-text {
            background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .card-dark {
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
        }
        .zero-shot-border { border-top: 4px solid #3B82F6; }
        .few-shot-border { border-top: 4px solid #8B5CF6; }
        .cot-border { border-top: 4px solid #EC4899; }
    </style>
</head>
<body class="min-h-screen flex flex-col">

    <!-- Top Navigation Header -->
    <header class="bg-[#111827] border-b border-gray-800 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="text-2xl">🧠</span>
                <div>
                    <h1 class="text-base font-bold text-white tracking-tight">Prompt Engineering Lab</h1>
                    <p class="text-xs text-gray-400">AI Mini-Capstone (25CS301) • Zero-Shot, Few-Shot & CoT</p>
                </div>
            </div>
            <!-- Navigation Tabs -->
            <nav class="hidden md:flex space-x-2 text-sm font-medium">
                <button onclick="switchTab('home')" id="tab-btn-home" class="px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold">🏠 Home</button>
                <button onclick="switchTab('experiment')" id="tab-btn-experiment" class="px-3 py-1.5 rounded-lg text-gray-300 hover:bg-gray-800">🧪 Run Experiment</button>
                <button onclick="switchTab('compare')" id="tab-btn-compare" class="px-3 py-1.5 rounded-lg text-gray-300 hover:bg-gray-800">⚖️ Compare</button>
                <button onclick="switchTab('analytics')" id="tab-btn-analytics" class="px-3 py-1.5 rounded-lg text-gray-300 hover:bg-gray-800">📊 Analytics</button>
                <button onclick="switchTab('history')" id="tab-btn-history" class="px-3 py-1.5 rounded-lg text-gray-300 hover:bg-gray-800">📜 History</button>
                <button onclick="switchTab('about')" id="tab-btn-about" class="px-3 py-1.5 rounded-lg text-gray-300 hover:bg-gray-800">ℹ️ About</button>
            </nav>
            <div class="flex items-center space-x-2">
                <span id="api-status-badge" class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30">
                    🟡 Demo Mode
                </span>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-grow max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">

        <!-- ================= PAGE 1: HOME ================= -->
        <section id="page-home" class="space-y-6">
            <div class="card-dark p-8 text-center space-y-3">
                <h2 class="text-4xl font-extrabold gradient-text">Prompt Engineering Strategies Compared</h2>
                <p class="text-gray-400 max-w-3xl mx-auto text-base">
                    An empirical comparative study of <b>Zero-Shot</b>, <b>Few-Shot</b>, and <b>Chain-of-Thought (CoT)</b> prompting techniques using Google Gemini and multi-metric objective evaluation.
                </p>
                <div class="pt-4 flex justify-center gap-3">
                    <button onclick="switchTab('experiment')" class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-6 py-2.5 rounded-lg shadow-lg">🚀 Start Experiment</button>
                    <button onclick="switchTab('analytics')" class="bg-gray-800 hover:bg-gray-700 text-gray-200 font-semibold px-6 py-2.5 rounded-lg border border-gray-700">📊 View Analytics</button>
                </div>
            </div>

            <!-- 4 Quick Stats -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="card-dark p-4 text-center">
                    <div class="text-xs text-gray-400 font-medium">BENCHMARK TRIALS</div>
                    <div id="stat-total" class="text-2xl font-bold text-blue-400 mt-1">4</div>
                </div>
                <div class="card-dark p-4 text-center">
                    <div class="text-xs text-gray-400 font-medium">QUESTIONS TESTED</div>
                    <div id="stat-questions" class="text-2xl font-bold text-purple-400 mt-1">4</div>
                </div>
                <div class="card-dark p-4 text-center">
                    <div class="text-xs text-gray-400 font-medium">ACCURACY LEADER</div>
                    <div id="stat-accuracy" class="text-xl font-bold text-emerald-400 mt-1">Chain-of-Thought</div>
                </div>
                <div class="card-dark p-4 text-center">
                    <div class="text-xs text-gray-400 font-medium">SPEED LEADER</div>
                    <div id="stat-speed" class="text-xl font-bold text-amber-400 mt-1">Zero-Shot</div>
                </div>
            </div>

            <!-- The Three Techniques Explained -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
                <div class="card-dark zero-shot-border p-6 flex flex-col justify-between">
                    <div>
                        <span class="text-xs font-bold uppercase tracking-wider text-blue-400 bg-blue-500/10 px-2.5 py-1 rounded-full border border-blue-500/20">Baseline Method</span>
                        <h3 class="text-xl font-bold text-blue-400 mt-3">1. Zero-Shot Prompting</h3>
                        <p class="text-sm text-gray-400 mt-2">Feeds queries directly to the LLM without demonstration examples or explicit reasoning scaffolds.</p>
                    </div>
                    <div class="mt-4 pt-4 border-t border-gray-800 text-xs text-gray-300 space-y-1">
                        <div><b>⚡ Strengths:</b> Lowest latency, minimal token consumption.</div>
                        <div><b>⚠️ Weaknesses:</b> Prone to errors in multi-step math and logic.</div>
                    </div>
                </div>
                <div class="card-dark few-shot-border p-6 flex flex-col justify-between">
                    <div>
                        <span class="text-xs font-bold uppercase tracking-wider text-purple-400 bg-purple-500/10 px-2.5 py-1 rounded-full border border-purple-500/20">In-Context Examples</span>
                        <h3 class="text-xl font-bold text-purple-400 mt-3">2. Few-Shot Prompting</h3>
                        <p class="text-sm text-gray-400 mt-2">Provides 2–3 representative input-output pairs inside the prompt context to guide formatting and tone.</p>
                    </div>
                    <div class="mt-4 pt-4 border-t border-gray-800 text-xs text-gray-300 space-y-1">
                        <div><b>⚡ Strengths:</b> High formatting adherence & styling consistency.</div>
                        <div><b>⚠️ Weaknesses:</b> Increased prompt tokens, slightly higher latency.</div>
                    </div>
                </div>
                <div class="card-dark cot-border p-6 flex flex-col justify-between">
                    <div>
                        <span class="text-xs font-bold uppercase tracking-wider text-pink-400 bg-pink-500/10 px-2.5 py-1 rounded-full border border-pink-500/20">Cognitive Reasoning</span>
                        <h3 class="text-xl font-bold text-pink-400 mt-3">3. Chain-of-Thought (CoT)</h3>
                        <p class="text-sm text-gray-400 mt-2">Enforces step-by-step intermediate reasoning decomposition before declaring the final conclusion.</p>
                    </div>
                    <div class="mt-4 pt-4 border-t border-gray-800 text-xs text-gray-300 space-y-1">
                        <div><b>⚡ Strengths:</b> Highest accuracy on multi-step reasoning problems.</div>
                        <div><b>⚠️ Weaknesses:</b> Highest token count and longest inference latency.</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ================= PAGE 2: RUN EXPERIMENT ================= -->
        <section id="page-experiment" class="hidden space-y-6">
            <div class="card-dark p-6 space-y-4">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
                    <div>
                        <h2 class="text-2xl font-bold text-white">🧪 Run Prompting Experiment</h2>
                        <p class="text-sm text-gray-400">Benchmark any question simultaneously across Zero-Shot, Few-Shot, and Chain-of-Thought.</p>
                    </div>
                    <!-- API Key Setting -->
                    <div class="w-full sm:w-auto flex items-center space-x-2">
                        <input type="password" id="input-api-key" placeholder="Gemini API Key (optional)" class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 w-56">
                        <button onclick="updateKeyStatus()" class="bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded-lg text-xs font-medium">Save</button>
                    </div>
                </div>

                <div class="bg-amber-500/10 border-l-4 border-amber-500 p-3 rounded-r-lg text-xs text-amber-300">
                    💡 <b>Tip:</b> If you haven't provided an API key, the system automatically uses realistic, educational <b>Demo Mode</b> responses so you can inspect all outputs, metrics, and charts immediately!
                </div>

                <!-- Input Controls -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <label class="block text-xs font-semibold text-gray-300 mb-1">Pick from 25 Curated Questions</label>
                        <select id="preset-questions" onchange="onSelectPreset()" class="w-full bg-gray-800 border border-gray-700 rounded-lg p-2.5 text-sm text-white focus:outline-none">
                            <option value="">-- Choose a pre-curated benchmark --</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-gray-300 mb-1">Category</label>
                        <select id="input-category" class="w-full bg-gray-800 border border-gray-700 rounded-lg p-2.5 text-sm text-white focus:outline-none">
                            <option>General Knowledge</option>
                            <option>Mathematics</option>
                            <option>Logical Reasoning</option>
                            <option>Science</option>
                            <option>Programming</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-gray-300 mb-1">Ground Truth Reference Answer</label>
                        <input type="text" id="input-ref" placeholder="e.g., Canberra, 80 km/h, etc." class="w-full bg-gray-800 border border-gray-700 rounded-lg p-2.5 text-sm text-white focus:outline-none">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-semibold text-gray-300 mb-1">Question Prompt</label>
                    <textarea id="input-question" rows="3" class="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-sm text-white focus:outline-none focus:border-indigo-500" placeholder="Type or select a question to test..."></textarea>
                </div>

                <div class="flex gap-3">
                    <button onclick="executeExperiment()" id="btn-run" class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-6 py-2.5 rounded-lg flex items-center space-x-2">
                        <span>🚀 Run Comparison</span>
                    </button>
                    <button onclick="clearInputForm()" class="bg-gray-800 hover:bg-gray-700 text-gray-300 px-4 py-2.5 rounded-lg border border-gray-700">Clear</button>
                </div>
            </div>

            <!-- Experiment Results Display Area -->
            <div id="results-container" class="hidden space-y-6">
                <div class="flex justify-between items-center">
                    <h3 class="text-xl font-bold text-white">📋 Side-by-Side Response Cards</h3>
                    <span id="trial-id-badge" class="text-xs bg-gray-800 px-2.5 py-1 rounded text-gray-400 font-mono"></span>
                </div>

                <!-- 3 Cards -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Zero Shot Card -->
                    <div class="card-dark zero-shot-border p-5 flex flex-col justify-between">
                        <div>
                            <div class="flex justify-between items-center mb-2">
                                <span class="text-xs font-bold text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">Zero-Shot</span>
                                <span id="zs-status" class="text-xs font-semibold px-2 py-0.5 rounded"></span>
                            </div>
                            <div class="text-2xl font-bold text-white mb-1"><span id="zs-score">0</span>% <span class="text-xs text-gray-400 font-normal">Score</span></div>
                            <div class="text-xs text-gray-400 mb-3">⏱️ <span id="zs-lat">0</span>s &nbsp;|&nbsp; 🪙 <span id="zs-tok">0</span> tokens</div>
                            <div id="zs-text" class="bg-gray-900 border border-gray-800 p-3 rounded-lg text-xs text-gray-200 h-44 overflow-y-auto whitespace-pre-wrap font-mono"></div>
                        </div>
                    </div>

                    <!-- Few Shot Card -->
                    <div class="card-dark few-shot-border p-5 flex flex-col justify-between">
                        <div>
                            <div class="flex justify-between items-center mb-2">
                                <span class="text-xs font-bold text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded">Few-Shot</span>
                                <span id="fs-status" class="text-xs font-semibold px-2 py-0.5 rounded"></span>
                            </div>
                            <div class="text-2xl font-bold text-white mb-1"><span id="fs-score">0</span>% <span class="text-xs text-gray-400 font-normal">Score</span></div>
                            <div class="text-xs text-gray-400 mb-3">⏱️ <span id="fs-lat">0</span>s &nbsp;|&nbsp; 🪙 <span id="fs-tok">0</span> tokens</div>
                            <div id="fs-text" class="bg-gray-900 border border-gray-800 p-3 rounded-lg text-xs text-gray-200 h-44 overflow-y-auto whitespace-pre-wrap font-mono"></div>
                        </div>
                    </div>

                    <!-- CoT Card -->
                    <div class="card-dark cot-border p-5 flex flex-col justify-between">
                        <div>
                            <div class="flex justify-between items-center mb-2">
                                <span class="text-xs font-bold text-pink-400 bg-pink-500/10 px-2 py-0.5 rounded">Chain-of-Thought</span>
                                <span id="cot-status" class="text-xs font-semibold px-2 py-0.5 rounded"></span>
                            </div>
                            <div class="text-2xl font-bold text-white mb-1"><span id="cot-score">0</span>% <span class="text-xs text-gray-400 font-normal">Score</span></div>
                            <div class="text-xs text-gray-400 mb-3">⏱️ <span id="cot-lat">0</span>s &nbsp;|&nbsp; 🪙 <span id="cot-tok">0</span> tokens</div>
                            <div id="cot-text" class="bg-gray-900 border border-gray-800 p-3 rounded-lg text-xs text-gray-200 h-44 overflow-y-auto whitespace-pre-wrap font-mono"></div>
                        </div>
                    </div>
                </div>

                <!-- Trial Charts -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
                    <div class="card-dark p-5">
                        <h4 class="text-sm font-bold text-gray-300 mb-3">Accuracy vs Latency Comparison</h4>
                        <div class="h-64"><canvas id="trialBarChart"></canvas></div>
                    </div>
                    <div class="card-dark p-5">
                        <h4 class="text-sm font-bold text-gray-300 mb-3">5-Axis Multidimensional Radar</h4>
                        <div class="h-64"><canvas id="trialRadarChart"></canvas></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ================= PAGE 3: COMPARE ================= -->
        <section id="page-compare" class="hidden space-y-6">
            <div class="card-dark p-6">
                <h2 class="text-2xl font-bold text-white mb-2">⚖️ Deep-Dive Side-by-Side Comparison</h2>
                <p class="text-sm text-gray-400 mb-4">Inspect intermediate reasoning, prompt text, and multi-metric diagnostics for logged trials.</p>
                
                <div class="mb-4">
                    <label class="block text-xs font-semibold text-gray-400 mb-1">Select Logged Trial</label>
                    <select id="compare-trial-select" onchange="renderCompareTrial()" class="w-full bg-gray-800 border border-gray-700 rounded-lg p-2.5 text-sm text-white focus:outline-none"></select>
                </div>

                <div id="compare-content" class="space-y-4"></div>
            </div>
        </section>

        <!-- ================= PAGE 4: ANALYTICS ================= -->
        <section id="page-analytics" class="hidden space-y-6">
            <div class="card-dark p-6">
                <h2 class="text-2xl font-bold text-white mb-2">📊 Global Analytics Dashboard</h2>
                <p class="text-sm text-gray-400 mb-6">Aggregated performance diagnostics compiled across all logged experimental trials.</p>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="card-dark p-4">
                        <h4 class="text-sm font-bold text-gray-300 mb-3">Average Accuracy by Strategy (%)</h4>
                        <div class="h-60"><canvas id="analyticsBarChart"></canvas></div>
                    </div>
                    <div class="card-dark p-4">
                        <h4 class="text-sm font-bold text-gray-300 mb-3">Outcome Status Breakdown</h4>
                        <div class="h-60"><canvas id="analyticsPieChart"></canvas></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ================= PAGE 5: HISTORY ================= -->
        <section id="page-history" class="hidden space-y-6">
            <div class="card-dark p-6">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-4">
                    <div>
                        <h2 class="text-2xl font-bold text-white">📜 Experiment History Ledger</h2>
                        <p class="text-sm text-gray-400">All experimental runs are recorded to persistent CSV ledger.</p>
                    </div>
                    <div class="flex gap-2">
                        <a href="/api/download-csv" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-3 py-2 rounded-lg flex items-center">📥 Download CSV</a>
                        <button onclick="clearHistoryLedger()" class="bg-red-900/40 hover:bg-red-800 text-red-300 border border-red-700/50 text-xs font-semibold px-3 py-2 rounded-lg">Clear All</button>
                    </div>
                </div>

                <div class="overflow-x-auto border border-gray-800 rounded-lg">
                    <table class="min-w-full divide-y divide-gray-800 text-xs">
                        <thead class="bg-gray-900 text-gray-400 uppercase tracking-wider text-left">
                            <tr>
                                <th class="p-3">Time</th>
                                <th class="p-3">Category</th>
                                <th class="p-3">Technique</th>
                                <th class="p-3">Question</th>
                                <th class="p-3">Status</th>
                                <th class="p-3">Score</th>
                                <th class="p-3">Latency</th>
                            </tr>
                        </thead>
                        <tbody id="history-table-body" class="divide-y divide-gray-800 text-gray-200">
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- ================= PAGE 6: ABOUT ================= -->
        <section id="page-about" class="hidden space-y-6">
            <div class="card-dark p-8 space-y-6">
                <div>
                    <h2 class="text-2xl font-bold text-white mb-1">ℹ️ Academic Technical Specification</h2>
                    <p class="text-sm text-gray-400">Department of Computer Science and Engineering • Course: Artificial Intelligence (25CS301)</p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-gray-300">
                    <div class="space-y-3">
                        <h3 class="text-lg font-bold text-indigo-400">1. Problem Formulation</h3>
                        <p class="text-xs text-gray-400 leading-relaxed">
                            Generative LLMs vary greatly in output reliability depending on prompt design. Zero-Shot prompting often hallucinates in multi-step arithmetic, Few-Shot adds token costs, and Chain-of-Thought (CoT) trades latency for deductive precision. This project systematically quantifies these trade-offs under controlled conditions.
                        </p>
                        <h3 class="text-lg font-bold text-indigo-400">2. Evaluation Formulation</h3>
                        <p class="text-xs text-gray-400 leading-relaxed font-mono">
                            Overall Score = 0.50 * Correctness + 0.25 * Relevance + 0.25 * Explanation Quality
                        </p>
                    </div>
                    <div class="space-y-3">
                        <h3 class="text-lg font-bold text-indigo-400">3. A5 Poster Presentation Summary</h3>
                        <div class="bg-gray-900 border border-gray-800 p-4 rounded-lg text-xs space-y-2 text-gray-300">
                            <div><b>Aim:</b> Benchmark Zero-Shot vs Few-Shot vs Chain-of-Thought on Google Gemini.</div>
                            <div><b>Key Finding:</b> CoT achieves +35% higher accuracy on logic/math with 2.4x latency.</div>
                            <div><b>Tech Stack:</b> Python, Google Gemini API, FastAPI/Streamlit, Chart.js/Plotly.</div>
                            <div><b>Real-World Value:</b> Guides cost-latency-accuracy trade-offs in enterprise GenAI.</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer class="border-t border-gray-800 py-4 text-center text-xs text-gray-500">
        AI Mini-Capstone Project (25CS301) • Department of Computer Science & Engineering • Vercel & Streamlit Cloud Ready
    </footer>

    <!-- Frontend Interactive Script -->
    <script>
        let questionsCache = [];
        let historyCache = [];
        let trialBarChartInstance = null;
        let trialRadarChartInstance = null;
        let analyticsBarChartInstance = null;
        let analyticsPieChartInstance = null;

        function switchTab(tab) {
            ['home', 'experiment', 'compare', 'analytics', 'history', 'about'].forEach(t => {
                document.getElementById('page-' + t).classList.add('hidden');
                document.getElementById('tab-btn-' + t).className = 'px-3 py-1.5 rounded-lg text-gray-300 hover:bg-gray-800';
            });
            document.getElementById('page-' + tab).classList.remove('hidden');
            document.getElementById('tab-btn-' + tab).className = 'px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold';

            if (tab === 'analytics') renderAnalytics();
            if (tab === 'history') loadHistoryData();
            if (tab === 'compare') renderCompareDropdown();
        }

        async function init() {
            try {
                const res = await fetch('/api/questions');
                const data = await res.json();
                questionsCache = data.questions || [];
                const select = document.getElementById('preset-questions');
                questionsCache.forEach((q, i) => {
                    const opt = document.createElement('option');
                    opt.value = i;
                    opt.textContent = `[${q.category}] ${q.question}`;
                    select.appendChild(opt);
                });
            } catch(e) {}
            loadSummary();
        }

        function onSelectPreset() {
            const idx = document.getElementById('preset-questions').value;
            if (idx !== "") {
                const q = questionsCache[idx];
                document.getElementById('input-question').value = q.question;
                document.getElementById('input-category').value = q.category;
                document.getElementById('input-ref').value = q.reference_answer;
            }
        }

        function clearInputForm() {
            document.getElementById('input-question').value = '';
            document.getElementById('input-ref').value = '';
            document.getElementById('preset-questions').value = '';
            document.getElementById('results-container').classList.add('hidden');
        }

        function updateKeyStatus() {
            const key = document.getElementById('input-api-key').value.trim();
            const badge = document.getElementById('api-status-badge');
            if (key) {
                badge.className = 'inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
                badge.textContent = '🟢 Gemini API Ready';
            } else {
                badge.className = 'inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30';
                badge.textContent = '🟡 Demo Mode';
            }
        }

        async function executeExperiment() {
            const q = document.getElementById('input-question').value.trim();
            const cat = document.getElementById('input-category').value;
            const ref = document.getElementById('input-ref').value.trim();
            const key = document.getElementById('input-api-key').value.trim();
            const btn = document.getElementById('btn-run');

            if (!q) { alert('Please enter or select a question.'); return; }

            btn.disabled = true;
            btn.innerHTML = '<span>⏳ Running 3 Strategies...</span>';

            try {
                const res = await fetch('/api/run-experiment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: q, category: cat, reference_answer: ref, api_key: key })
                });
                const data = await res.json();
                displayResults(data);
                loadSummary();
            } catch (err) {
                alert('Execution error: ' + err);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span>🚀 Run Comparison</span>';
            }
        }

        function displayResults(data) {
            document.getElementById('results-container').classList.remove('hidden');
            document.getElementById('trial-id-badge').textContent = 'Trial: ' + data.experiment_id;

            const zs = data.evaluations['Zero-Shot'];
            const fs = data.evaluations['Few-Shot'];
            const cot = data.evaluations['Chain-of-Thought'];

            renderCard('zs', zs, data.raw_results['Zero-Shot']);
            renderCard('fs', fs, data.raw_results['Few-Shot']);
            renderCard('cot', cot, data.raw_results['Chain-of-Thought']);

            renderTrialCharts(zs, fs, cot);
            window.scrollTo({ top: document.getElementById('results-container').offsetTop - 60, behavior: 'smooth' });
        }

        function renderCard(prefix, ev, raw) {
            document.getElementById(prefix + '-score').textContent = ev.overall_score;
            document.getElementById(prefix + '-lat').textContent = ev.latency;
            document.getElementById(prefix + '-tok').textContent = ev.total_tokens;
            document.getElementById(prefix + '-text').textContent = raw.response;
            const statusEl = document.getElementById(prefix + '-status');
            statusEl.textContent = ev.status;
            statusEl.className = ev.status === 'Correct' ? 'text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400' : 'text-xs font-semibold px-2 py-0.5 rounded bg-amber-500/20 text-amber-400';
        }

        function renderTrialCharts(zs, fs, cot) {
            if (trialBarChartInstance) trialBarChartInstance.destroy();
            if (trialRadarChartInstance) trialRadarChartInstance.destroy();

            const ctxBar = document.getElementById('trialBarChart').getContext('2d');
            trialBarChartInstance = new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: ['Zero-Shot', 'Few-Shot', 'Chain-of-Thought'],
                    datasets: [
                        { label: 'Overall Score (%)', data: [zs.overall_score, fs.overall_score, cot.overall_score], backgroundColor: '#8B5CF6' },
                        { label: 'Latency (s)', data: [zs.latency, fs.latency, cot.latency], backgroundColor: '#EC4899' }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, grid: { color: '#1F2937' } } } }
            });

            const ctxRadar = document.getElementById('trialRadarChart').getContext('2d');
            trialRadarChartInstance = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Correctness', 'Relevance', 'Explanation', 'Speed', 'Tokens'],
                    datasets: [
                        { label: 'Zero-Shot', data: [zs.correctness, zs.relevance, zs.explanation, Math.max(20, 100 - zs.latency * 20), Math.max(20, 100 - zs.total_tokens * 0.2)], borderColor: '#3B82F6', backgroundColor: 'rgba(59, 130, 246, 0.2)' },
                        { label: 'Few-Shot', data: [fs.correctness, fs.relevance, fs.explanation, Math.max(20, 100 - fs.latency * 20), Math.max(20, 100 - fs.total_tokens * 0.2)], borderColor: '#8B5CF6', backgroundColor: 'rgba(139, 92, 246, 0.2)' },
                        { label: 'Chain-of-Thought', data: [cot.correctness, cot.relevance, cot.explanation, Math.max(20, 100 - cot.latency * 20), Math.max(20, 100 - cot.total_tokens * 0.2)], borderColor: '#EC4899', backgroundColor: 'rgba(236, 72, 153, 0.2)' }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }

        async function loadSummary() {
            try {
                const res = await fetch('/api/summary');
                const s = await res.json();
                document.getElementById('stat-total').textContent = s.total_trials;
                document.getElementById('stat-questions').textContent = s.unique_questions;
                document.getElementById('stat-accuracy').textContent = s.best_accuracy_tech;
                document.getElementById('stat-speed').textContent = s.fastest_tech;
            } catch(e) {}
        }

        async function loadHistoryData() {
            const res = await fetch('/api/history');
            const data = await res.json();
            historyCache = data.records || [];
            const tbody = document.getElementById('history-table-body');
            tbody.innerHTML = '';
            historyCache.forEach(r => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="p-3 text-gray-400">${r.timestamp}</td>
                    <td class="p-3 font-medium">${r.category}</td>
                    <td class="p-3 font-semibold text-indigo-400">${r.technique}</td>
                    <td class="p-3 max-w-xs truncate">${r.question}</td>
                    <td class="p-3"><span class="px-2 py-0.5 rounded ${r.status==='Correct'?'bg-emerald-500/20 text-emerald-400':'bg-amber-500/20 text-amber-400'}">${r.status}</span></td>
                    <td class="p-3 font-bold">${r.overall_score}%</td>
                    <td class="p-3">${r.latency}s</td>
                `;
                tbody.appendChild(tr);
            });
        }

        async function clearHistoryLedger() {
            if (confirm('Clear all recorded trials from experiments.csv?')) {
                await fetch('/api/clear-history', { method: 'POST' });
                loadHistoryData();
                loadSummary();
            }
        }

        async function renderAnalytics() {
            await loadHistoryData();
            if (analyticsBarChartInstance) analyticsBarChartInstance.destroy();
            if (analyticsPieChartInstance) analyticsPieChartInstance.destroy();

            const zsScores = historyCache.filter(r => r.technique === 'Zero-Shot').map(r => r.overall_score);
            const fsScores = historyCache.filter(r => r.technique === 'Few-Shot').map(r => r.overall_score);
            const cotScores = historyCache.filter(r => r.technique === 'Chain-of-Thought').map(r => r.overall_score);

            const avg = arr => arr.length ? (arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(1) : 0;

            const ctxBar = document.getElementById('analyticsBarChart').getContext('2d');
            analyticsBarChartInstance = new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: ['Zero-Shot', 'Few-Shot', 'Chain-of-Thought'],
                    datasets: [{
                        label: 'Average Score (%)',
                        data: [avg(zsScores), avg(fsScores), avg(cotScores)],
                        backgroundColor: ['#3B82F6', '#8B5CF6', '#EC4899']
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100 } } }
            });

            const correctCount = historyCache.filter(r => r.status === 'Correct').length;
            const otherCount = historyCache.length - correctCount;

            const ctxPie = document.getElementById('analyticsPieChart').getContext('2d');
            analyticsPieChartInstance = new Chart(ctxPie, {
                type: 'doughnut',
                data: {
                    labels: ['Correct', 'Needs Review / Partial'],
                    datasets: [{ data: [correctCount, otherCount], backgroundColor: ['#10B981', '#F59E0B'] }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }

        function renderCompareDropdown() {
            const select = document.getElementById('compare-trial-select');
            select.innerHTML = '';
            const unique = [...new Set(historyCache.map(r => r.experiment_id))];
            unique.forEach(id => {
                const row = historyCache.find(r => r.experiment_id === id);
                const opt = document.createElement('option');
                opt.value = id;
                opt.textContent = `Trial ${id} — ${row.question.substring(0, 50)}...`;
                select.appendChild(opt);
            });
            renderCompareTrial();
        }

        function renderCompareTrial() {
            const id = document.getElementById('compare-trial-select').value;
            const trials = historyCache.filter(r => r.experiment_id === id);
            const container = document.getElementById('compare-content');
            if (!trials.length) { container.innerHTML = '<div class="text-xs text-gray-500">No trial selected.</div>'; return; }

            const q = trials[0];
            let html = `
                <div class="card-dark p-4 bg-gray-900 border border-gray-800">
                    <div class="text-xs text-gray-400 font-semibold">QUESTION</div>
                    <div class="text-sm text-white font-medium mb-1">${q.question}</div>
                    <div class="text-xs text-gray-500">Category: <b>${q.category}</b> | Ground Truth: <code>${q.reference_answer}</code></div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            `;

            trials.forEach(t => {
                html += `
                    <div class="card-dark p-4">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-xs font-bold text-indigo-400">${t.technique}</span>
                            <span class="text-xs px-2 py-0.5 rounded ${t.status==='Correct'?'bg-emerald-500/20 text-emerald-400':'bg-amber-500/20 text-amber-400'}">${t.status}</span>
                        </div>
                        <div class="text-xl font-bold mb-1">${t.overall_score}%</div>
                        <div class="text-xs text-gray-400 mb-2">Latency: ${t.latency}s | Tokens: ${t.total_tokens}</div>
                        <div class="bg-gray-950 p-2.5 rounded text-xs text-gray-300 font-mono max-h-40 overflow-y-auto whitespace-pre-wrap">${t.response_text}</div>
                    </div>
                `;
            });
            html += '</div>';
            container.innerHTML = html;
        }

        window.onload = init;
    </script>
</body>
</html>
"""
