"""
Visualization Module for Prompt Engineering Comparison
Generates modern, interactive Plotly charts and performance dashboards:
1. Accuracy & Overall Score Bar Chart
2. Latency / Response Time Bar Chart
3. 5-Axis Multidimensional Radar Chart
4. Correctness Distribution Donut / Pie Chart
5. Category-wise Performance Grouped Bar Chart
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
from config import TECHNIQUE_COLORS


def plot_accuracy_comparison(eval_df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive bar chart comparing Correctness and Overall Score
    for Zero-Shot, Few-Shot, and Chain-of-Thought prompting.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=eval_df["technique"],
        y=eval_df["correctness"],
        name="Correctness Score",
        marker_color="#3B82F6",
        text=eval_df["correctness"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Correctness: %{y:.1f}%<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=eval_df["technique"],
        y=eval_df["overall_score"],
        name="Overall Composite Score",
        marker_color="#8B5CF6",
        text=eval_df["overall_score"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Overall Score: %{y:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Accuracy & Overall Score Comparison</b>",
        barmode="group",
        yaxis=dict(title="Score (0 - 100)", range=[0, 115]),
        xaxis=dict(title="Prompting Technique"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        height=380
    )
    return fig


def plot_latency_comparison(eval_df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive bar chart comparing Response Generation Latency in seconds.
    """
    colors = [TECHNIQUE_COLORS.get(t, "#3B82F6") for t in eval_df["technique"]]

    fig = go.Figure(go.Bar(
        x=eval_df["technique"],
        y=eval_df["latency"],
        marker_color=colors,
        text=eval_df["latency"].apply(lambda v: f"{v:.2f}s"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Response Time: %{y:.3f} seconds<extra></extra>"
    ))

    max_lat = eval_df["latency"].max() if not eval_df.empty else 1.0
    fig.update_layout(
        title="<b>Response Generation Latency (Seconds)</b>",
        yaxis=dict(title="Latency (s)", range=[0, max_lat * 1.25 + 0.1]),
        xaxis=dict(title="Prompting Technique"),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        height=380
    )
    return fig


def plot_radar_performance(eval_df: pd.DataFrame) -> go.Figure:
    """
    Creates a 5-dimensional Radar/Spider chart comparing:
    - Correctness
    - Relevance
    - Explanation Quality
    - Speed (Normalized latency)
    - Token Efficiency
    """
    categories = [
        "Correctness",
        "Relevance",
        "Explanation Quality",
        "Speed (Latency)",
        "Token Efficiency"
    ]

    fig = go.Figure()

    for _, row in eval_df.iterrows():
        tech = row["technique"]
        corr = float(row.get("correctness", 0))
        rel = float(row.get("relevance", 0))
        exp = float(row.get("explanation", 0))
        
        # Normalize speed: lower latency = higher score (capped between 20 and 100)
        lat = float(row.get("latency", 1.0))
        speed_score = max(20.0, min(100.0, 100.0 - (lat * 20.0)))
        
        # Token efficiency: fewer tokens = higher efficiency score
        tok = float(row.get("total_tokens", 50))
        token_eff = max(20.0, min(100.0, 100.0 - (tok * 0.2)))

        values = [corr, rel, exp, speed_score, token_eff]
        values.append(values[0])  # Close the radar loop

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=tech,
            line=dict(color=TECHNIQUE_COLORS.get(tech, "#3B82F6"), width=2),
            opacity=0.65
        ))

    fig.update_layout(
        title="<b>Multidimensional Performance Radar</b>",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9)
            )
        ),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=60, b=60),
        height=420
    )
    return fig


def plot_status_pie(history_df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive Donut chart showing distribution of correctness outcomes.
    """
    if history_df.empty or "status" not in history_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No experiment data available yet.", showarrow=False, font=dict(size=14))
        fig.update_layout(template="plotly_dark", height=350)
        return fig

    status_counts = history_df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    color_map = {
        "Correct": "#10B981",          # Green
        "Partially Correct": "#F59E0B",# Amber
        "Mostly Correct": "#3B82F6",   # Blue
        "Incorrect": "#EF4444",        # Red
        "Needs Review": "#EC4899",     # Pink
        "Failed / Error": "#6B7280"    # Gray
    }
    colors = [color_map.get(s, "#8B5CF6") for s in status_counts["status"]]

    fig = go.Figure(data=[go.Pie(
        labels=status_counts["status"],
        values=status_counts["count"],
        hole=0.45,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value} (%{percent})<extra></extra>"
    )])

    fig.update_layout(
        title="<b>Evaluation Outcome Distribution</b>",
        template="plotly_dark",
        margin=dict(l=30, r=30, t=60, b=30),
        height=350
    )
    return fig


def plot_category_performance(history_df: pd.DataFrame) -> go.Figure:
    """
    Plots average overall score by question category across prompting techniques.
    """
    if history_df.empty or "category" not in history_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No historical experiments found.", showarrow=False, font=dict(size=14))
        fig.update_layout(template="plotly_dark", height=380)
        return fig

    grouped = (
        history_df.groupby(["category", "technique"])["overall_score"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        grouped,
        x="category",
        y="overall_score",
        color="technique",
        barmode="group",
        color_discrete_map=TECHNIQUE_COLORS,
        labels={"overall_score": "Average Score (%)", "category": "Question Category", "technique": "Technique"},
        title="<b>Average Performance by Question Category</b>"
    )

    fig.update_layout(
        template="plotly_dark",
        yaxis=dict(range=[0, 110]),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_token_usage_breakdown(eval_df: pd.DataFrame) -> go.Figure:
    """
    Plots stacked bar chart of prompt vs completion tokens across techniques.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=eval_df["technique"],
        y=eval_df["prompt_tokens"],
        name="Prompt Tokens",
        marker_color="#6366F1",
        text=eval_df["prompt_tokens"],
        textposition="inside"
    ))

    fig.add_trace(go.Bar(
        x=eval_df["technique"],
        y=eval_df["completion_tokens"],
        name="Completion Tokens",
        marker_color="#EC4899",
        text=eval_df["completion_tokens"],
        textposition="inside"
    ))

    fig.update_layout(
        title="<b>Token Consumption Breakdown</b>",
        barmode="stack",
        yaxis=dict(title="Total Tokens"),
        xaxis=dict(title="Prompting Technique"),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380
    )
    return fig
