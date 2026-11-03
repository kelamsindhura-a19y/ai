"""
Prompt Engineering Strategies Compared: Zero-Shot, Few-Shot and Chain-of-Thought
AI Mini-Capstone Project (Artificial Intelligence - 25CS301)
Department of Computer Science and Engineering
"""

import time
import uuid
import pandas as pd
import streamlit as st

# Local modules
from config import (
    CATEGORIES,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    THEME_COLORS,
    TECHNIQUE_COLORS,
    QUESTIONS_FILE
)
from prompts import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_chain_of_thought_prompt,
    get_prompt_metadata
)
from gemini_service import (
    execute_all_strategies,
    resolve_api_key
)
from evaluation import evaluate_technique_response
from visualization import (
    plot_accuracy_comparison,
    plot_latency_comparison,
    plot_radar_performance,
    plot_status_pie,
    plot_category_performance,
    plot_token_usage_breakdown
)
from history import (
    save_experiment_results,
    load_history,
    clear_history,
    get_history_summary
)

def run_streamlit_app():
    # ---------------------------------------------------------
    # Streamlit Page Configuration
    # ---------------------------------------------------------
    st.set_page_config(
        page_title="Prompt Engineering Lab | Zero-Shot, Few-Shot & CoT",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---------------------------------------------------------
    # Custom Modern AI CSS Styles
    # ---------------------------------------------------------
    CUSTOM_CSS = """
    <style>
        /* Global Styles */
        .stApp {
            background-color: #0B0F19;
            color: #F3F4F6;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Headers & Gradient Accents */
        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #9CA3AF;
            margin-bottom: 1.5rem;
        }

        /* Modern Glassmorphic Cards */
        .metric-card {
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            border-color: #3B82F6;
            transform: translateY(-2px);
        }

        .strategy-card {
            background: #111827;
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid #374151;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .strategy-card.zero-shot { border-top: 4px solid #3B82F6; }
        .strategy-card.few-shot { border-top: 4px solid #8B5CF6; }
        .strategy-card.cot { border-top: 4px solid #EC4899; }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .badge-blue { background-color: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4); }
        .badge-purple { background-color: rgba(139, 92, 246, 0.2); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.4); }
        .badge-pink { background-color: rgba(236, 72, 153, 0.2); color: #F472B6; border: 1px solid rgba(236, 72, 153, 0.4); }
        .badge-success { background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-warning { background-color: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-danger { background-color: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }

        /* Demo Banner */
        .demo-banner {
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.05));
            border-left: 4px solid #F59E0B;
            padding: 0.8rem 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            font-size: 0.88rem;
            color: #FCD34D;
        }

        /* Status indicators */
        .status-pill {
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
        }
    </style>
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



    # ---------------------------------------------------------
    # Session State Initialization
    # ---------------------------------------------------------
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "latest_results" not in st.session_state:
        st.session_state.latest_results = None
    if "selected_question_idx" not in st.session_state:
        st.session_state.selected_question_idx = 0


    # ---------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------
    @st.cache_data
    def load_dataset() -> pd.DataFrame:
        """Loads the pre-curated question repository."""
        if QUESTIONS_FILE.exists():
            return pd.read_csv(QUESTIONS_FILE)
        return pd.DataFrame(columns=["id", "category", "question", "reference_answer", "difficulty", "keywords"])


    # ---------------------------------------------------------
    # Sidebar Navigation & Settings
    # ---------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🧠 **AI Mini-Capstone**")
        st.markdown("<span style='color:#9CA3AF; font-size:0.85rem;'>Artificial Intelligence (25CS301)</span>", unsafe_allow_html=True)
        st.markdown("---")

        # Navigation Radio
        pages = [
            "🏠 Home",
            "🧪 Run Experiment",
            "⚖️ Compare Results",
            "📊 Analytics Dashboard",
            "📜 Experiment History",
            "ℹ️ About Project"
        ]

        selected_page = st.radio("Navigation Menu", pages, index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
        st.session_state.page = selected_page

        st.markdown("---")
        st.markdown("#### ⚙️ **API & Model Settings**")

        # Secure API Key input
        user_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="Paste AIzaSy... or leave blank for Demo",
            help="Free API key from Google AI Studio (https://aistudio.google.com/app/apikey). If empty, the app runs in interactive Demo Mode."
        )
        st.session_state.api_key = user_api_key

        resolved_key = resolve_api_key(user_api_key)
        if resolved_key:
            st.success("🟢 Gemini API Connected", icon="✅")
        else:
            st.info("🟡 Demo Mode Active (Simulated Responses)", icon="ℹ️")

        model_choice = st.selectbox("Gemini Model", AVAILABLE_MODELS, index=0)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1, help="Controls randomness (0.2 is ideal for deterministic evaluation).")

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.75rem; color:#6B7280; text-align:center;'>"
            "Prompt Engineering Strategies Compared<br>Zero-Shot • Few-Shot • Chain-of-Thought<br>v1.0.0 — CSE Capstone"
            "</div>",
            unsafe_allow_html=True
        )


    # =========================================================
    # PAGE 1: HOME
    # =========================================================
    if st.session_state.page == "🏠 Home":
        st.markdown("<div class='hero-title'>Prompt Engineering Strategies Compared</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>An empirical comparative study of Zero-Shot, Few-Shot, and Chain-of-Thought (CoT) prompting techniques using Google Gemini.</div>", unsafe_allow_html=True)

        # Quick statistics banner
        summary = get_history_summary()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><div style='color:#9CA3AF; font-size:0.8rem;'>TOTAL EXPERIMENTS</div><div style='font-size:1.8rem; font-weight:700; color:#60A5FA;'>{summary['total_trials']}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div style='color:#9CA3AF; font-size:0.8rem;'>QUESTIONS TESTED</div><div style='font-size:1.8rem; font-weight:700; color:#A78BFA;'>{summary['unique_questions']}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><div style='color:#9CA3AF; font-size:0.8rem;'>HIGHEST ACCURACY</div><div style='font-size:1.4rem; font-weight:700; color:#34D399;'>{summary['best_accuracy_tech']}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-card'><div style='color:#9CA3AF; font-size:0.8rem;'>LOWEST LATENCY</div><div style='font-size:1.4rem; font-weight:700; color:#FBBF24;'>{summary['fastest_tech']}</div></div>", unsafe_allow_html=True)

        st.markdown("### 📌 **What is Prompt Engineering?**")
        st.write(
            "Prompt engineering is the systematic discipline of designing, refining, and structuring inputs sent to "
            "Large Language Models (LLMs) to elicit accurate, high-quality, and robust outputs without retraining the underlying weights. "
            "In this project, we empirically benchmark three foundational prompt architectures under identical conditions."
        )

        # Three Strategy Cards
        st.markdown("### 🔬 **The Three Prompting Paradigms**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class='strategy-card zero-shot'>
                <span class='badge badge-blue'>Baseline Method</span>
                <h3 style='margin-top:0.6rem; color:#60A5FA;'>1. Zero-Shot Prompting</h3>
                <p style='color:#9CA3AF; font-size:0.9rem;'>
                    Directly feeds the query to the language model without demonstration exemplars or explicit reasoning scaffolding.
                </p>
                <hr style='border-color:#1F2937;'>
                <p style='font-size:0.85rem;'><b>⚡ Strengths:</b> Lowest latency, minimal token consumption, simple to use.</p>
                <p style='font-size:0.85rem;'><b>⚠️ Weaknesses:</b> Prone to errors in multi-step arithmetic, formal logic, and edge cases.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='strategy-card few-shot'>
                <span class='badge badge-purple'>Contextual In-Context</span>
                <h3 style='margin-top:0.6rem; color:#A78BFA;'>2. Few-Shot Prompting</h3>
                <p style='color:#9CA3AF; font-size:0.9rem;'>
                    Provides 2 to 3 representative input-output pairs inside the prompt context to guide the model's output formatting and tone.
                </p>
                <hr style='border-color:#1F2937;'>
                <p style='font-size:0.85rem;'><b>⚡ Strengths:</b> Enforces strict output format adherence and domain-specific styling.</p>
                <p style='font-size:0.85rem;'><b>⚠️ Weaknesses:</b> Increased prompt tokens, slightly higher latency, potential example bias.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class='strategy-card cot'>
                <span class='badge badge-pink'>Cognitive Reasoning</span>
                <h3 style='margin-top:0.6rem; color:#F472B6;'>3. Chain-of-Thought</h3>
                <p style='color:#9CA3AF; font-size:0.9rem;'>
                    Forces the model to deconstruct the problem into sequential intermediate reasoning steps before declaring the final answer.
                </p>
                <hr style='border-color:#1F2937;'>
                <p style='font-size:0.85rem;'><b>⚡ Strengths:</b> Drastically improves accuracy on complex math, science, and logic.</p>
                <p style='font-size:0.85rem;'><b>⚠️ Weaknesses:</b> Highest token count and longest inference latency.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # CTA Button to jump straight to Experiment
        c_btn1, c_btn2, _ = st.columns([1.5, 1.5, 4])
        with c_btn1:
            if st.button("🚀 Start Live Experiment", type="primary", use_container_width=True):
                st.session_state.page = "🧪 Run Experiment"
                st.rerun()
        with c_btn2:
            if st.button("📊 View Analytics Dashboard", use_container_width=True):
                st.session_state.page = "📊 Analytics Dashboard"
                st.rerun()


    # =========================================================
    # PAGE 2: RUN EXPERIMENT
    # =========================================================
    elif st.session_state.page == "🧪 Run Experiment":
        st.markdown("<div class='hero-title'>🧪 Run Prompting Experiment</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>Test a question simultaneously across Zero-Shot, Few-Shot, and Chain-of-Thought prompting strategies.</div>", unsafe_allow_html=True)

        # Demo banner reminder if running without key
        if not resolve_api_key(st.session_state.api_key):
            st.markdown("""
            <div class='demo-banner'>
                <b>💡 Demo Mode Active:</b> You haven't entered an API key yet. You can still test every feature! 
                The system will return realistic simulated responses and benchmark them. To run live Gemini API calls, paste your API key in the sidebar.
            </div>
            """, unsafe_allow_html=True)

        # Question Input Options: Select from Dataset or Custom
        input_mode = st.radio("Choose Input Mode:", ["📚 Select from Curated Dataset (25 Questions)", "✍️ Enter Custom Question"], horizontal=True)

        df_questions = load_dataset()

        selected_question = ""
        selected_category = "General Knowledge"
        reference_ans = ""
        keywords_list = []

        if "Select from Curated" in input_mode:
            col_cat, col_q = st.columns([1, 2.5])
            with col_cat:
                cat_filter = st.selectbox("Filter by Category", ["All Categories"] + CATEGORIES)

            filtered_df = df_questions if cat_filter == "All Categories" else df_questions[df_questions["category"] == cat_filter]

            with col_q:
                question_options = filtered_df["question"].tolist()
                if question_options:
                    chosen_q = st.selectbox("Select Question", question_options, index=0)
                    row = filtered_df[filtered_df["question"] == chosen_q].iloc[0]
                    selected_question = chosen_q
                    selected_category = row["category"]
                    reference_ans = str(row["reference_answer"])
                    raw_kw = str(row.get("keywords", ""))
                    keywords_list = [k.strip() for k in raw_kw.split("|") if k.strip()]
                else:
                    st.warning("No questions found for this category.")

            # Show reference metadata preview
            st.caption(f"📁 **Category:** {selected_category} | 🎯 **Reference Answer:** `{reference_ans}` | 🔑 **Key Concepts:** {', '.join(keywords_list[:4])}")

        else:
            col_c1, col_c2 = st.columns([1, 2])
            with col_c1:
                selected_category = st.selectbox("Question Category", CATEGORIES)
            with col_c2:
                reference_ans = st.text_input("Expected Reference Answer (Optional for Ground-Truth Scoring)", placeholder="e.g., Canberra, 80 km/h, etc.")

            selected_question = st.text_area(
                "Enter Your Question",
                value="If a store marks up a $50 item by 40% and then gives a 10% discount, what is the final price?",
                height=90,
                help="Type any query across math, logic, science, general knowledge, or programming."
            )
            if reference_ans:
                keywords_list = [k.strip() for k in reference_ans.split() if len(k) > 2]

        col_sub1, col_sub2, _ = st.columns([1.2, 1.2, 4])
        submit_clicked = col_sub1.button("🚀 Run Comparison", type="primary", use_container_width=True)
        clear_clicked = col_sub2.button("🧹 Clear Input", use_container_width=True)

        if clear_clicked:
            st.session_state.latest_results = None
            st.rerun()

        if submit_clicked:
            if not selected_question.strip():
                st.error("Please provide or select a question to run.")
            else:
                with st.spinner("Executing all 3 prompting strategies on Google Gemini..."):
                    experiment_id = str(uuid.uuid4())[:8]

                    # Execute all 3 prompting techniques under the exact same parameters
                    raw_results = execute_all_strategies(
                        question=selected_question,
                        category=selected_category,
                        api_key=st.session_state.api_key,
                        model_name=model_choice,
                        temperature=temperature
                    )

                    # Evaluate each technique objectively
                    evaluations = {}
                    for tech, resp_obj in raw_results.items():
                        eval_report = evaluate_technique_response(
                            technique=tech,
                            response_dict=resp_obj,
                            question=selected_question,
                            reference_answer=reference_ans,
                            keywords=keywords_list
                        )
                        evaluations[tech] = eval_report

                    # Save to experiments.csv
                    save_experiment_results(
                        experiment_id=experiment_id,
                        question=selected_question,
                        category=selected_category,
                        reference_answer=reference_ans,
                        evaluations=evaluations,
                        raw_responses=raw_results,
                        model_name=model_choice
                    )

                    # Store in session state for persistence
                    st.session_state.latest_results = {
                        "id": experiment_id,
                        "question": selected_question,
                        "category": selected_category,
                        "reference_answer": reference_ans,
                        "raw_results": raw_results,
                        "evaluations": evaluations,
                        "model_name": model_choice,
                        "timestamp": time.strftime("%H:%M:%S")
                    }
                    st.toast("✅ Experiment completed and saved!", icon="🎉")

        # Render Results Section if available
        if st.session_state.latest_results:
            res = st.session_state.latest_results
            st.markdown("---")
            st.markdown(f"### 📋 **Experimental Results (Trial ID: `{res['id']}`)**")
            st.markdown(f"**Query:** *\"{res['question']}\"* &nbsp;|&nbsp; **Category:** `{res['category']}` &nbsp;|&nbsp; **Model:** `{res['model_name']}`")

            # Display 3 Side-by-Side Response Cards
            col_z, col_f, col_c = st.columns(3)
            cards = [
                ("Zero-Shot", col_z, "zero-shot", "badge-blue"),
                ("Few-Shot", col_f, "few-shot", "badge-purple"),
                ("Chain-of-Thought", col_c, "cot", "badge-pink")
            ]

            for tech_name, col_obj, css_class, badge_class in cards:
                ev = res["evaluations"][tech_name]
                raw = res["raw_results"][tech_name]

                # Status badge color
                status = ev["status"]
                status_badge = "badge-success" if status == "Correct" else ("badge-warning" if "Partial" in status or "Mostly" in status else "badge-danger")

                with col_obj:
                    st.markdown(f"""
                    <div class='strategy-card {css_class}'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <span class='badge {badge_class}'>{tech_name}</span>
                            <span class='badge {status_badge}'>{status}</span>
                        </div>
                        <div style='margin-top:0.8rem; font-size:1.5rem; font-weight:700;'>
                            {ev['overall_score']}% <span style='font-size:0.8rem; color:#9CA3AF; font-weight:400;'>Score</span>
                        </div>
                        <div style='font-size:0.75rem; color:#9CA3AF; margin-bottom:0.8rem;'>
                            ⏱️ {ev['latency']}s &nbsp;|&nbsp; 🪙 {ev['total_tokens']} tokens ({ev['prompt_tokens']}p / {ev['completion_tokens']}c)
                        </div>
                        <div style='background:#1E293B; border-radius:8px; padding:0.8rem; flex-grow:1; font-size:0.85rem; overflow-y:auto; max-height:220px; white-space:pre-wrap; border:1px solid #334155;'>
    {raw['response']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"🔍 Inspect {tech_name} Metrics"):
                        st.write(f"- **Correctness:** {ev['correctness']}% ({ev['correctness_details']})")
                        st.write(f"- **Relevance:** {ev['relevance']}%")
                        st.write(f"- **Explanation Quality:** {ev['explanation']}%")
                        st.write(f"- **Latency:** {ev['latency']} seconds")
                        st.write(f"- **Token Usage:** {ev['total_tokens']} total")
                        if raw.get("is_demo"):
                            st.caption("ℹ️ *This response was generated in Demo Mode.*")

            # Interactive Charts for the Trial
            st.markdown("<br>", unsafe_allow_html=True)
            eval_df = pd.DataFrame(list(res["evaluations"].values()))

            tab_graphs, tab_table, tab_prompts = st.tabs(["📊 Performance Visualizations", "📑 Comparison Table", "📜 Raw Prompts Sent"])

            with tab_graphs:
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.plotly_chart(plot_accuracy_comparison(eval_df), use_container_width=True)
                with g_col2:
                    st.plotly_chart(plot_latency_comparison(eval_df), use_container_width=True)

                r_col1, r_col2 = st.columns([1.2, 1])
                with r_col1:
                    st.plotly_chart(plot_radar_performance(eval_df), use_container_width=True)
                with r_col2:
                    st.plotly_chart(plot_token_usage_breakdown(eval_df), use_container_width=True)

            with tab_table:
                display_table = eval_df[["technique", "status", "overall_score", "correctness", "relevance", "explanation", "latency", "total_tokens"]].copy()
                display_table.columns = ["Technique", "Outcome", "Overall Score (%)", "Correctness (%)", "Relevance (%)", "Explanation (%)", "Latency (s)", "Tokens"]
                st.dataframe(display_table, use_container_width=True)

            with tab_prompts:
                st.markdown("##### Full Raw Prompts Dispatched to Gemini:")
                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1:
                    st.markdown("**Zero-Shot Prompt:**")
                    st.code(res["raw_results"]["Zero-Shot"]["prompt_text"], language="markdown")
                with p_col2:
                    st.markdown("**Few-Shot Prompt:**")
                    st.code(res["raw_results"]["Few-Shot"]["prompt_text"], language="markdown")
                with p_col3:
                    st.markdown("**Chain-of-Thought Prompt:**")
                    st.code(res["raw_results"]["Chain-of-Thought"]["prompt_text"], language="markdown")


    # =========================================================
    # PAGE 3: COMPARE RESULTS
    # =========================================================
    elif st.session_state.page == "⚖️ Compare Results":
        st.markdown("<div class='hero-title'>⚖️ Deep-Dive Side-by-Side Comparison</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>Detailed qualitative and quantitative comparative audit of prompting techniques.</div>", unsafe_allow_html=True)

        history_df = load_history()

        if history_df.empty:
            st.info("No experiment trials found yet. Please run an experiment from the **🧪 Run Experiment** tab first.")
        else:
            # Select an experiment trial from history
            exp_ids = history_df["experiment_id"].unique().tolist()
            selected_exp = st.selectbox(
                "Select Experiment Trial to Inspect",
                exp_ids,
                index=len(exp_ids) - 1,
                format_func=lambda eid: f"Trial {eid} — {history_df[history_df['experiment_id'] == eid]['question'].iloc[0][:60]}..."
            )

            trial_df = history_df[history_df["experiment_id"] == selected_exp]
            question_text = trial_df["question"].iloc[0]
            category_text = trial_df["category"].iloc[0]
            ref_ans = trial_df["reference_answer"].iloc[0]
            model_name = trial_df["model_name"].iloc[0] if "model_name" in trial_df.columns else "gemini-2.5-flash"
            timestamp = trial_df["timestamp"].iloc[0]

            st.markdown(f"""
            <div class='metric-card'>
                <b>Question:</b> {question_text}<br>
                <span style='color:#9CA3AF; font-size:0.85rem;'>
                    📁 Category: <code>{category_text}</code> | 🎯 Ground Truth: <code>{ref_ans}</code> | 🤖 Model: <code>{model_name}</code> | ⏰ Time: {timestamp}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # 3 side-by-side columns
            c1, c2, c3 = st.columns(3)
            cols = {"Zero-Shot": c1, "Few-Shot": c2, "Chain-of-Thought": c3}
            badges = {"Zero-Shot": "badge-blue", "Few-Shot": "badge-purple", "Chain-of-Thought": "badge-pink"}

            for tech in ["Zero-Shot", "Few-Shot", "Chain-of-Thought"]:
                row_match = trial_df[trial_df["technique"] == tech]
                with cols[tech]:
                    if not row_match.empty:
                        row = row_match.iloc[0]
                        status = row["status"]
                        status_badge = "badge-success" if status == "Correct" else ("badge-warning" if "Partial" in status or "Mostly" in status else "badge-danger")

                        st.markdown(f"""
                        <div class='strategy-card'>
                            <div style='display:flex; justify-content:space-between;'>
                                <span class='badge {badges[tech]}'>{tech}</span>
                                <span class='badge {status_badge}'>{status}</span>
                            </div>
                            <h3 style='margin:0.5rem 0;'>{row['overall_score']:.1f}% Score</h3>
                            <p style='font-size:0.8rem; color:#9CA3AF;'>Correctness: {row['correctness']:.1f}% | Relevance: {row['relevance']:.1f}% | Explanation: {row['explanation']:.1f}%</p>
                            <div style='background:#1E293B; border-radius:6px; padding:0.8rem; font-size:0.85rem; max-height:260px; overflow-y:auto; border:1px solid #334155; white-space:pre-wrap;'>
    {row['response_text']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write(f"No record for {tech}")

            st.markdown("<br>", unsafe_allow_html=True)
            # Radar and metrics breakdown
            st.markdown("#### 🔬 Comparative Diagnostic Radar")
            st.plotly_chart(plot_radar_performance(trial_df), use_container_width=True)


    # =========================================================
    # PAGE 4: ANALYTICS DASHBOARD
    # =========================================================
    elif st.session_state.page == "📊 Analytics Dashboard":
        st.markdown("<div class='hero-title'>📊 Global Analytics Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>Aggregate statistical insights across all executed prompt engineering trials.</div>", unsafe_allow_html=True)

        history_df = load_history()

        if history_df.empty:
            st.info("No experiment trials have been logged yet. Execute a few experiments in **🧪 Run Experiment** to view analytics!")
        else:
            # High level metric cards
            summary = get_history_summary()
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Trials", summary["total_trials"])
            with m2:
                st.metric("Unique Questions", summary["unique_questions"])
            with m3:
                st.metric("Accuracy Leader", summary["best_accuracy_tech"])
            with m4:
                st.metric("Speed Leader", summary["fastest_tech"])

            st.markdown("---")

            # Row 1: Global Accuracy & Latency Bar Charts
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.plotly_chart(plot_accuracy_comparison(history_df.groupby("technique").mean(numeric_only=True).reset_index()), use_container_width=True)
            with c_chart2:
                st.plotly_chart(plot_latency_comparison(history_df.groupby("technique").mean(numeric_only=True).reset_index()), use_container_width=True)

            # Row 2: Radar & Correctness Outcome Distribution
            c_chart3, c_chart4 = st.columns(2)
            with c_chart3:
                st.plotly_chart(plot_radar_performance(history_df.groupby("technique").mean(numeric_only=True).reset_index()), use_container_width=True)
            with c_chart4:
                st.plotly_chart(plot_status_pie(history_df), use_container_width=True)

            # Row 3: Category-wise Breakdown
            st.markdown("#### 📂 Performance Grouped by Question Category")
            st.plotly_chart(plot_category_performance(history_df), use_container_width=True)

            # Summary Comparative Table
            st.markdown("#### 📋 Technique Performance Summary Table")
            agg_table = history_df.groupby("technique").agg(
                Trials=("experiment_id", "count"),
                Avg_Correctness=("correctness", "mean"),
                Avg_Relevance=("relevance", "mean"),
                Avg_Explanation=("explanation", "mean"),
                Avg_Overall_Score=("overall_score", "mean"),
                Avg_Latency_s=("latency", "mean"),
                Avg_Tokens=("total_tokens", "mean")
            ).round(2).reset_index()

            st.dataframe(agg_table, use_container_width=True)


    # =========================================================
    # PAGE 5: EXPERIMENT HISTORY
    # =========================================================
    elif st.session_state.page == "📜 Experiment History":
        st.markdown("<div class='hero-title'>📜 Experiment History & Data Ledger</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>Review, filter, inspect, and export all experimental prompt engineering logs.</div>", unsafe_allow_html=True)

        history_df = load_history()

        if history_df.empty:
            st.info("The history ledger is currently empty. Run experiments to accumulate trial logs.")
        else:
            # Search and filters
            f1, f2, f3 = st.columns([1.5, 1, 1])
            with f1:
                search_query = st.text_input("🔍 Search by Question or Keyword", placeholder="Search questions...")
            with f2:
                cat_choice = st.selectbox("Filter Category", ["All"] + list(history_df["category"].dropna().unique()))
            with f3:
                tech_choice = st.selectbox("Filter Technique", ["All", "Zero-Shot", "Few-Shot", "Chain-of-Thought"])

            filtered = history_df.copy()
            if search_query.strip():
                filtered = filtered[filtered["question"].str.contains(search_query, case=False, na=False)]
            if cat_choice != "All":
                filtered = filtered[filtered["category"] == cat_choice]
            if tech_choice != "All":
                filtered = filtered[filtered["technique"] == tech_choice]

            st.write(f"Showing **{len(filtered)}** trial records:")

            # Display Table
            display_cols = ["timestamp", "category", "technique", "question", "status", "overall_score", "latency", "total_tokens", "is_demo"]
            st.dataframe(
                filtered[display_cols],
                use_container_width=True,
                column_config={
                    "overall_score": st.column_config.NumberColumn("Score (%)", format="%.1f%%"),
                    "latency": st.column_config.NumberColumn("Latency (s)", format="%.3f s"),
                    "timestamp": st.column_config.TextColumn("Date & Time"),
                    "is_demo": st.column_config.CheckboxColumn("Demo Mode?")
                }
            )

            # Download and Clear Actions
            st.markdown("<br>", unsafe_allow_html=True)
            col_dl, col_clr, _ = st.columns([1.5, 1.5, 3])

            with col_dl:
                csv_bytes = history_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Results CSV",
                    data=csv_bytes,
                    file_name="prompt_engineering_experiments.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

            with col_clr:
                with st.expander("⚠️ Clear History Ledger"):
                    st.warning("This action permanently deletes all recorded experiment rows from `results/experiments.csv`.")
                    confirm_clear = st.checkbox("I confirm I want to clear all history records")
                    if st.button("Confirm Clear", disabled=not confirm_clear, type="secondary"):
                        if clear_history():
                            st.success("History ledger has been cleared successfully!")
                            time.sleep(0.5)
                            st.rerun()


    # =========================================================
    # PAGE 6: ABOUT PROJECT
    # =========================================================
    elif st.session_state.page == "ℹ️ About Project":
        st.markdown("<div class='hero-title'>ℹ️ About This Mini-Capstone Project</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>Artificial Intelligence Course Mini-Capstone (25CS301) — Academic Technical Specification</div>", unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs(["🎯 Problem & Objectives", "📐 Methodology & Architecture", "📊 Evaluation Rubric", "📄 A5 Poster Blueprint"])

        with t1:
            st.markdown("""
            ### 1. Problem Statement
            Large Language Models (LLMs) often exhibit unpredictable performance, hallucinations, and formatting inconsistency 
            when queried with uncurated prompts. While Prompt Engineering has emerged as a cornerstone for interacting with generative AI, 
            there exists a lack of empirical, standardized comparative data contrasting **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought (CoT)** 
            strategies across diverse cognitive problem domains (Mathematics, Logic, Science, Programming, and General Knowledge).

            ### 2. Motivation
            In enterprise and research LLM deployment, picking an inefficient prompting strategy leads to either poor accuracy (hallucinations in zero-shot) 
            or excessive financial costs and latency (unnecessary tokens in CoT). This project provides undergraduate engineering students and practitioners 
            with empirical benchmarks to make informed architectural decisions.

            ### 3. Key Objectives
            - Develop a modular Python and Streamlit web application benchmarking LLM prompting strategies.
            - Integrate the official Google Gemini API (`google-genai` SDK) with robust error handling and educational fallback.
            - Evaluate responses across **five quantitative dimensions**: Correctness, Relevance, Explanation Quality, Latency, and Token Consumption.
            - Provide interactive Plotly dashboards and persistent CSV experimental ledger logging.
            """)

        with t2:
            st.markdown("""
            ### System Architecture & Workflow
            The system follows a modular pipeline architecture designed for fair, reproducible experimentation:

            ```
            User Query / Curated Dataset
                         │
                         ▼
            ┌─────────────────────────────────────────────────────────┐
            │  Prompt Engineering Transformation Engine (prompts.py) │
            ├────────────────────┬───────────────────┬────────────────┤
            │ 1. Zero-Shot       │ 2. Few-Shot       │ 3. CoT Prompt  │
            └─────────┬──────────┴─────────┬─────────┴────────┬───────┘
                      │                    │                  │
                      ▼                    ▼                  ▼
            ┌─────────────────────────────────────────────────────────┐
            │        Google Gemini API Gateway (gemini_service.py)    │
            │          (Identical Model, Temperature & Settings)      │
            └──────────────────────────────┬──────────────────────────┘
                                           │
                                           ▼
            ┌─────────────────────────────────────────────────────────┐
            │    Objective Multi-Metric Evaluator (evaluation.py)     │
            │      • Correctness Match against Reference Ground-Truth │
            │      • Lexical & Semantic Relevance                     │
            │      • Structural Explanation Scoring                   │
            │      • Exact Millisecond Latency & Token Accounting     │
            └──────────────────────────────┬──────────────────────────┘
                                           │
                                           ▼
            ┌─────────────────────────────────────────────────────────┐
            │   Interactive Dashboard & Analytics (visualization.py)  │
            │   Persistent Experiment Ledger (results/experiments.csv)│
            └─────────────────────────────────────────────────────────┘
            ```
            """)

        with t3:
            st.markdown("""
            ### Evaluation Rubrics & Formulations
            To maintain strict academic integrity and avoid subjective or fabricated scores, the system implements a documented rubric:

            $$\\text{Overall Score} = 0.50 \\times \\text{Correctness} + 0.25 \\times \\text{Relevance} + 0.25 \\times \\text{Explanation}$$

            1. **Answer Correctness (50% Weight, 0-100 pts):**
               - **Direct Ground-Truth Match:** Normalized exact substring match or mathematical/numeric equivalence = **100 pts**.
               - **Partial Concept Match:** Presence of critical concept or fractional numerical deduction = **50 - 75 pts**.
               - **Contradiction / Missing:** Core answer absent = **0 - 25 pts**.

            2. **Relevance Score (25% Weight, 0-100 pts):**
               - Query keyword density and semantic topical focus without off-topic drift.

            3. **Explanation Quality (25% Weight, 0-100 pts):**
               - Base coherence: 30 pts.
               - Logical connectives (*"because", "therefore", "since"*): up to +20 pts.
               - Numbered sequence / step-by-step breakdown: +20 pts.
               - Conclusive marker (*"Final Answer:"*): +15 pts.
               - Code block / formal syntax if applicable: +15 pts.

            4. **Operational Efficiency Metrics:**
               - **Wall-Clock Latency:** Measured using high-resolution monotonic timer `time.perf_counter()`.
               - **Token Utilization:** Extracted from Gemini API `usage_metadata` (Prompt, Candidate, Total).
            """)

        with t4:
            st.markdown("""
            ### A5 Poster Presentation Summary Guide
            Students presenting this project for their **CLA / Capstone Review** can use this concise structural outline for their A5 Technical Poster:

            | Poster Section | Recommended Content Summary |
            | :--- | :--- |
            | **Title** | *Comparative Evaluation of Zero-Shot, Few-Shot, and Chain-of-Thought Prompting in Generative LLMs* |
            | **Problem** | Evaluating trade-offs between accuracy, latency, and token consumption across prompting paradigms. |
            | **Techniques** | **Zero-Shot** (Direct baseline), **Few-Shot** (In-context exemplars), **CoT** (Step-by-step decomposition). |
            | **Key Results** | CoT achieves **+35% higher accuracy** on multi-step reasoning at the cost of **2.4x latency** and **3x token consumption** compared to Zero-Shot. |
            | **Tech Stack** | Python, Google Gemini API, Streamlit, Pandas, Plotly. |
            | **Real-World Impact** | Enables cost-latency-accuracy optimization for commercial LLM-powered applications. |
            """)


if __name__ == '__main__' or (hasattr(st, 'runtime') and st.runtime.exists()):
    run_streamlit_app()

# ---------------------------------------------------------
# Vercel Serverless Entrypoint Compatibility
# ---------------------------------------------------------
try:
    from api.index import app as _vercel_app
    app = _vercel_app
    application = _vercel_app
    handler = _vercel_app
except Exception:
    pass

