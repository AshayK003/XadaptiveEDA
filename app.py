import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import time
import logging
import uuid

from data_processor import DataProcessor
from recommendation_engine import RecommendationEngine
from preference_learner import PreferenceTracker
from visualization_generator import VisualizationGenerator
from insight_generator import explain_recommendation, compare_recommendations, global_explanation_summary
from constants import DEFAULT_PREFERENCES, ANALYSIS_GOALS
from data_quality import QualityReport
import llm_adapter
import session_persistence as sp

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
log = logging.getLogger('app')


def _compute_epsilon():
    return 0.1 if st.session_state.get('_exploration_mode') else 0.0

def _compute_col_affinity():
    """Compute per-column affinity score (mean Jaccard vs all previously selected columns)."""
    freq = st.session_state.get('_col_frequency', {})
    cooc = st.session_state.get('_col_cooccurrence', {})
    if not freq:
        return {}
    affinity = {}
    for col_a in freq:
        total_j = 0.0
        n = 0
        for col_b in freq:
            if col_a >= col_b:
                continue
            both = cooc.get(col_a, {}).get(col_b, 0)
            if both == 0:
                continue
            jaccard = both / (freq[col_a] + freq[col_b] - both)
            total_j += jaccard
            n += 1
        affinity[col_a] = total_j / n if n > 0 else 0.0
    return affinity


# Page configuration
st.set_page_config(
    page_title="Adaptive EDA System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve app appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stButton>button:focus-visible,
    .stCheckbox>label:focus-visible {
        outline: 2px solid #1f77b4;
        outline-offset: 2px;
    }
    div[data-testid="stExpander"] details summary p {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for persistence
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = DEFAULT_PREFERENCES.copy()
    
if 'interaction_history' not in st.session_state:
    st.session_state.interaction_history = []

if 'data_profile' not in st.session_state:
    st.session_state.data_profile = None
    
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

if 'df' not in st.session_state:
    st.session_state.df = None

if 'quality_report' not in st.session_state:
    st.session_state.quality_report = None

if '_active_goal' not in st.session_state:
    st.session_state._active_goal = None

if '_expert_mode' not in st.session_state:
    st.session_state._expert_mode = False

if '_viewed_combos' not in st.session_state:
    st.session_state._viewed_combos = set()

if '_viz_start_time' not in st.session_state:
    st.session_state._viz_start_time = {}

if '_col_frequency' not in st.session_state:
    st.session_state._col_frequency = {}

if '_col_cooccurrence' not in st.session_state:
    st.session_state._col_cooccurrence = {}

if '_exploration_mode' not in st.session_state:
    st.session_state._exploration_mode = False

if '_session_id' not in st.session_state:
    st.session_state._session_id = str(uuid.uuid4())[:8]

# Initialize components
data_processor = DataProcessor()
recommendation_engine = RecommendationEngine()
preference_learner = PreferenceTracker()
preference_learner.preference_weights = st.session_state.user_preferences
visualization_generator = VisualizationGenerator()

# App title and description
st.title("🔍 Exploratory Data Analysis Assistant")
st.markdown("### Profiles your data, recommends analyses, generates visualizations")

# Sidebar for dataset upload and preferences
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a dataset file", type=["csv", "xlsx", "xls", "json"])
    
    if uploaded_file is not None:
        file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size / 1024:.1f} KB"}
        st.write("File Details:", file_details)
        if uploaded_file.size > 50 * 1024 * 1024:
            st.warning("Large file (>50 MB) — processing may be slow or fail.")
    
    st.header("Your Priorities")
    
    # Analysis goals
    st.markdown("#### Analysis Goal")
    goal_options = [None] + list(ANALYSIS_GOALS.keys())
    goal_labels = ["None (custom)"] + list(ANALYSIS_GOALS.keys())
    current_goal_idx = goal_options.index(st.session_state.get('_active_goal')) if st.session_state.get('_active_goal') in goal_options else 0
    selected_goal = st.radio(
        "Focus",
        options=goal_options,
        format_func=lambda x: goal_labels[goal_options.index(x)] if x in goal_options else "None (custom)",
        index=current_goal_idx,
        key="_goal_radio"
    )
    if selected_goal != st.session_state.get('_active_goal'):
        st.session_state._active_goal = selected_goal
        if selected_goal:
            preference_learner.set_goal(selected_goal)
        else:
            preference_learner.set_goal(None)
        st.session_state.user_preferences = preference_learner.preference_weights
        if st.session_state.data_profile is not None:
            st.session_state.recommendations = recommendation_engine.generate_recommendations(
                st.session_state.data_profile, st.session_state.user_preferences,
                quality_score=getattr(st.session_state.quality_report, 'overall_quality_score', None) if st.session_state.get('_finalized') else None,
                viewed_combos=st.session_state._viewed_combos,
                interaction_history=st.session_state.interaction_history,
                col_affinity=_compute_col_affinity(),
                epsilon=_compute_epsilon()
            )
        st.rerun()
    
    # Save / Load / Decay
    col_s, col_l = st.columns(2)
    with col_s:
        if st.button("Save Preferences"):
            preference_learner.save_preferences()
            st.toast("Saved to ~/.eda_assistant_prefs.json", icon="💾")
    with col_l:
        if st.button("Load Preferences"):
            ok = preference_learner.load_preferences()
            if ok:
                st.session_state.user_preferences = preference_learner.preference_weights
                st.rerun()
            else:
                st.toast("No saved preferences found", icon="❌")
    if st.button("Apply Time Decay"):
        preference_learner.apply_temporal_decay()
        st.session_state.user_preferences = preference_learner.preference_weights
        st.toast("Decay applied — older interactions lose influence", icon="⏳")
    
    st.markdown("#### Priority Levels")
    if st.session_state.user_preferences:
        prefs = st.session_state.user_preferences
        pref_df = pd.DataFrame([
            {'type': k.replace('_', ' ').title(), 'score': v}
            for k, v in prefs.items()
        ]).sort_values('score', ascending=True)

        fig = px.bar(
            pref_df, x='score', y='type', orientation='h',
            text='score', range_x=[0, 1.1],
            color_discrete_sequence=['#1f77b4'],
            height=200
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(
            showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
            xaxis_visible=False, yaxis_title=None
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with st.expander("Adjust Priorities", expanded=False):
            st.write("Set how much each analysis type is weighted in recommendations:")
            with st.form(key="preference_form"):
                new_preferences = {}
                for analysis_type, current_value in st.session_state.user_preferences.items():
                    friendly_name = analysis_type.replace('_', ' ').title()
                    new_preferences[analysis_type] = st.slider(
                        f"{friendly_name}",
                        min_value=0.1,
                        max_value=1.0,
                        value=float(current_value),
                        step=0.1,
                        format="%.1f"
                    )
                submit_button = st.form_submit_button(label="Update Preferences")
                if submit_button:
                    st.session_state.user_preferences = preference_learner.set_preferences(new_preferences)
                    if st.session_state.data_profile is not None:
                        st.session_state.recommendations = recommendation_engine.generate_recommendations(
                            st.session_state.data_profile,
                            st.session_state.user_preferences,
                            viewed_combos=st.session_state._viewed_combos,
                            interaction_history=st.session_state.interaction_history,
                            col_affinity=_compute_col_affinity()
                        )
                    st.toast("Preferences updated — recommendations reordered.", icon="✅")
    
    # Reset preferences button
    if st.button("Reset Preferences"):
        st.session_state.user_preferences = DEFAULT_PREFERENCES.copy()
        preference_learner.preference_weights = st.session_state.user_preferences
        st.session_state.interaction_history = []
        st.toast("Preferences reset to defaults.", icon="🔄")
    
    st.header("Session")
    st.session_state._exploration_mode = st.toggle("Exploration mode (ε-greedy)", value=st.session_state.get('_exploration_mode', False), help="Occasionally shows lower-ranked analyses to discover new insights")
    with st.expander("Save / Load Session", expanded=False):
        session_name = st.text_input("Session name", value=f"session_{st.session_state._session_id}", key="_session_name_input")
        col_s, col_l = st.columns(2)
        with col_s:
            if st.button("Save Session"):
                sp.save_session(
                    session_name,
                    st.session_state.user_preferences,
                    st.session_state.interaction_history,
                    active_goal=st.session_state.get('_active_goal'),
                    last_file=st.session_state.get('last_file'),
                    profile_json=st.session_state.data_profile
                )
                st.toast(f"Saved session: {session_name}", icon="💾")
        with col_l:
            if st.button("Load Session"):
                data = sp.load_session(session_name)
                if data:
                    st.session_state.user_preferences = data["preferences"]
                    preference_learner.preference_weights = data["preferences"]
                    st.session_state.interaction_history = data["interaction_history"]
                    st.session_state._active_goal = data.get("active_goal")
                    if data.get("last_file"):
                        st.session_state.last_file = data["last_file"]
                    if data.get("profile_json"):
                        st.session_state.data_profile = data["profile_json"]
                    st.toast(f"Loaded session: {session_name}", icon="📂")
                    st.rerun()
                else:
                    st.toast(f"Session '{session_name}' not found", icon="❌")
        sessions = sp.list_sessions()
        if sessions:
            st.caption("Recent sessions:")
            for s in sessions:
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.caption(f"{s['id']} — {s['file'] or 'no file'} ({s['updated']})")
                with c2:
                    if st.button("Delete", key=f"_del_{s['id']}"):
                        sp.delete_session(s['id'])
                        st.rerun()
    
    st.header("AI Analysis")
    st.session_state._expert_mode = st.toggle("Expert Mode", value=st.session_state.get('_expert_mode', False), help="Show raw data, JSON details, and download options")
    
    st.session_state.ai_enabled = st.toggle(
        "Enable AI insights",
        value=st.session_state.get('ai_enabled', True),
        help="Use a local model (Ollama) or a free remote API"
    )
    st.session_state._chat_enabled = st.toggle(
        "Ask anything about your data (chat)",
        value=st.session_state.get('_chat_enabled', False),
        help="Chat with the LLM about your dataset — works with the same provider"
    )
    if st.session_state.ai_enabled or st.session_state.get('_chat_enabled'):
        env_provider = os.environ.get("LLM_PROVIDER", "local")
        provider = st.selectbox(
            "Provider",
            options=["local", "openrouter", "groq", "custom"],
            format_func=lambda x: {"local": "Local (Ollama)", "openrouter": "OpenRouter (free tier)",
                                   "groq": "Groq (free tier)", "custom": "Custom API"}[x],
            index=["local", "openrouter", "groq", "custom"].index(env_provider) if env_provider in ["local", "openrouter", "groq", "custom"] else 0,
            key="_ai_provider"
        )
        # Clear stale local status when switching away from local
        if provider != "local" and st.session_state.get('_llm_status') is not None:
            st.session_state._llm_status = None
        if provider == "local":
            status = st.session_state.get('_llm_status')
            if status is None:
                status = llm_adapter.check_ollama()
                st.session_state._llm_status = status
            if status.get("ok"):
                models = status["models"]
                active_model = llm_adapter.pick_model(models)
                st.session_state._llm_model = active_model
                st.caption(f" Using {active_model}")
                if active_model not in models:
                    st.caption(f"Pull it: `ollama pull {active_model}`")
            else:
                st.session_state._llm_model = None
                st.session_state._llm_endpoint = None
                st.caption("Ollama not running — install from ollama.ai")
        else:
            env_key_map = {"openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY"}
            api_key = os.environ.get(env_key_map.get(provider, ""), "")
            env_model = os.environ.get("LLM_MODEL", "")
            ep = llm_adapter.PROVIDERS.get(provider, {}).get("endpoint", "")
            if provider == "custom":
                ep = st.text_input("Endpoint URL", value=ep, key="_ai_endpoint")
                api_key = os.environ.get("CUSTOM_API_KEY", "")
            if api_key and (ep or provider != "custom"):
                st.session_state._llm_endpoint = ep
                model_from_env = env_model if env_model else llm_adapter.PROVIDERS.get(provider, {}).get("default_model", "")
                st.session_state._llm_model = model_from_env
                st.caption(f" {provider.capitalize()} configured via .env")
            else:
                st.session_state._llm_endpoint = None
                st.session_state._llm_model = None
                st.caption(f"Set {env_key_map.get(provider, 'CUSTOM_API_KEY')} in .env")
    else:
        st.session_state._llm_status = None
        st.session_state._llm_endpoint = None
        st.session_state._llm_model = None

# Main content area
if uploaded_file is not None:
    # Load and process data if not already done
    if st.session_state.df is None or uploaded_file.name != getattr(st.session_state, 'last_file', ''):
        try:
            progress = st.progress(0, text="Reading file...")
            df = data_processor.load_data(uploaded_file)

            progress.progress(15, text="Cleaning and validating data...")
            df, quality_report = data_processor.cleanse(df)
            st.session_state.quality_report = quality_report

            st.session_state.df = df
            st.session_state.last_file = uploaded_file.name
            st.session_state.pop('_column_names', None)

            # Progressive stratified sampling for large datasets
            if len(df) > 50000:
                st.info(f"Large dataset ({len(df):,} rows). Sampling can speed up exploration.")
                if st.checkbox("Sample to ~10,000 rows (stratified)", key="sample_large_dataset",
                               help="Preserves categorical distributions, keeps rare patterns"):
                    df = data_processor.sample_stratified(df, target_rows=10000)
                    st.session_state.df = df

            progress.progress(35, text="Profiling dataset...")
            data_profile = data_processor.profile_dataset(df)
            st.session_state.data_profile = data_profile
            
            # Cache column cardinalities for the recommendation loop
            st.session_state._cardinality = {col: df[col].nunique() for col in df.columns}
            
            progress.progress(70, text="Generating recommendations...")
            recommendations = recommendation_engine.generate_recommendations(
                data_profile, 
                st.session_state.user_preferences,
                viewed_combos=st.session_state._viewed_combos,
                interaction_history=st.session_state.interaction_history,
                col_affinity=_compute_col_affinity(),
                epsilon=_compute_epsilon()
            )
            st.session_state.recommendations = recommendations
            progress.progress(100, text="Done")
            progress.empty()
        except Exception as e:
            st.error(f"Could not process **{uploaded_file.name}**: {str(e)}")
            with st.expander("Technical details"):
                st.code(
                    f"Error type: {type(e).__name__}\n"
                    f"File: {uploaded_file.name}\n"
                    f"Extension: {os.path.splitext(uploaded_file.name)[1].lower()}\n"
                    f"Python: {sys.executable}"
                )
            st.stop()
    else:
        # Use cached data
        df = st.session_state.df
        data_profile = st.session_state.data_profile
        recommendations = st.session_state.recommendations

    # Unpack quality report
    quality_report = st.session_state.get('quality_report')
    if quality_report is None:
        quality_report = QualityReport()

    # Unnamed column renaming
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols and '_cols_renamed' not in st.session_state:
        st.subheader("Unnamed Columns Detected")
        st.write(f"Found {len(unnamed_cols)} column(s) without names.")
        
        # Data preview toggle for unnamed columns
        show_preview = st.checkbox("Preview unnamed column data", key="_preview_unnamed")
        if show_preview:
            preview_cols = unnamed_cols + [c for c in df.columns if c not in unnamed_cols][:3]
            st.dataframe(df[preview_cols].head(5), use_container_width=True)
        
        if st.session_state.get('ai_enabled') and '_column_names' not in st.session_state:
            with st.spinner("Suggesting column names from data..."):
                provider = st.session_state.get('_ai_provider', 'local')
                model = st.session_state.get('_llm_model') or llm_adapter.DEFAULT_MODEL
                endpoint = st.session_state.get('_llm_endpoint')
                if not endpoint and provider != "local":
                    st.info("Configure an API key and endpoint in the sidebar to use AI naming suggestions.")
                    st.session_state._column_names = {}
                else:
                    key_map = {"openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY", "custom": "CUSTOM_API_KEY"}
                    api_key = os.environ.get(key_map.get(provider, ""), "")
                    result = llm_adapter.suggest_column_names(
                        df, unnamed_cols, model=model, provider=provider,
                        api_key=api_key, endpoint=endpoint or ""
                    )
                    if result.get("ok"):
                        st.session_state._column_names = result["names"]
                        st.success(f"Suggested {len(result['names'])} column name(s)")
                    else:
                        st.warning(f"AI naming unavailable: {result.get('error')}")
                        st.session_state._column_names = {}
            if st.session_state.get('_column_names'):
                st.caption("AI-suggested names are pre-filled below")

        suggested = st.session_state.get('_column_names') or {}
        rename_map = {}
        cols = st.columns(min(3, len(unnamed_cols)))
        for i, col in enumerate(unnamed_cols):
            with cols[i % len(cols)]:
                default_val = suggested.get(col, col)
                new_name = st.text_input(f"Rename `{col}`", value=default_val, key=f"_rename_{i}")
                if new_name and new_name != col:
                    rename_map[col] = new_name

        if rename_map:
            if st.button("Apply Column Renames"):
                df.rename(columns=rename_map, inplace=True)
                st.session_state.df = df
                st.session_state._cols_renamed = True
                st.rerun()
    elif not unnamed_cols and '_cols_renamed' not in st.session_state:
        st.session_state._cols_renamed = True
        st.rerun()

    # Drop first N rows (after column rename, before finalize)
    if '_cols_renamed' in st.session_state and '_rows_dropped' not in st.session_state and '_finalized' not in st.session_state:
        with st.expander("Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        drop_rows = st.number_input("Drop first N rows (optional)", min_value=0, max_value=len(df)-1, value=0, step=1, key="_drop_rows")
        if drop_rows > 0:
            if st.button(f"Remove first {drop_rows} row(s)"):
                df.drop(index=df.index[:drop_rows], inplace=True)
                df.reset_index(drop=True, inplace=True)
                st.session_state.df = df
                st.session_state._rows_dropped = drop_rows
                st.rerun()

    # Finalize: update all analytics and clear AI cache
    if '_cols_renamed' in st.session_state and '_finalized' not in st.session_state:
        with st.expander("Final Data Preview (pre-finalize)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        if st.button("Finalize Dataset — Generate Full Analysis"):
            st.session_state.df = df
            st.session_state.data_profile = data_processor.profile_dataset(df)
            st.session_state._cardinality = {col: df[col].nunique() for col in df.columns}
            st.session_state.quality_report = data_processor.cleanse(df, skip_name_normalization=True)[1]
            st.session_state.recommendations = recommendation_engine.generate_recommendations(
                st.session_state.data_profile, st.session_state.user_preferences,
                quality_score=st.session_state.quality_report.overall_quality_score,
                viewed_combos=st.session_state._viewed_combos,
                interaction_history=st.session_state.interaction_history,
                col_affinity=_compute_col_affinity(),
                epsilon=_compute_epsilon()
            )
            for k in list(st.session_state.keys()):
                if k.startswith('_ai_'):
                    del st.session_state[k]
            st.session_state._finalized = True
            st.rerun()

    if '_finalized' in st.session_state:
        dropped = st.session_state.get('_rows_dropped', 0)
        parts = []
        if '_cols_renamed' in st.session_state:
            parts.append("Columns renamed")
        if dropped:
            parts.append(f"Dropped {dropped} row(s)")
        if parts:
            st.caption(" | ".join(parts))
        st.divider()
        
        # ── Chat with your data ────────────────────────────────
        if st.session_state.get('_chat_enabled'):
            st.markdown("#### 💬 Ask anything about your data")
            if '_chat_history' not in st.session_state:
                st.session_state._chat_history = []
            for msg in st.session_state._chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            chat_input = st.chat_input("Ask a question about your dataset...", key="_data_chat")
            if chat_input:
                st.session_state._chat_history.append({"role": "user", "content": chat_input})
                provider = st.session_state.get('_ai_provider', 'local')
                model = st.session_state.get('_llm_model', llm_adapter.DEFAULT_MODEL)
                endpoint = st.session_state.get('_llm_endpoint')
                key_map = {"openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY", "custom": "CUSTOM_API_KEY"}
                api_key = os.environ.get(key_map.get(provider, ""), "")
                result = llm_adapter.chat_with_data(
                    chat_input, data_profile, df,
                    conversation_history=st.session_state._chat_history[:-1],
                    provider=provider, model=model, endpoint=endpoint, api_key=api_key
                )
                if result.get("ok"):
                    st.session_state._chat_history.append({"role": "assistant", "content": result["text"]})
                else:
                    st.session_state._chat_history.append({"role": "assistant", "content": f"⚠️ {result.get('error', 'LLM unavailable')}"})
                st.rerun()
        
        # ── Global Exploration Summary ─────────────────────────
        if st.session_state.interaction_history:
            with st.expander("Exploration Summary", expanded=False):
                st.markdown(global_explanation_summary(
                    data_profile, quality_report,
                    st.session_state.interaction_history,
                    st.session_state.user_preferences
                ))
    
    # ── Quality warnings ─────────────────────────────────────
    if quality_report.warnings:
        for w in quality_report.warnings[:5]:
            st.warning(w)
        if len(quality_report.warnings) > 5:
            with st.expander(f"+{len(quality_report.warnings) - 5} more warnings"):
                for w in quality_report.warnings[5:]:
                    st.warning(w)

    # Only show full analysis after finalization
    if '_finalized' not in st.session_state:
        st.info("Complete the steps above and click **Finalize Dataset** to see analysis.")
        st.stop()

    # Display data overview
    st.subheader("Data Overview")
    
    # Show shape and sample
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        total_missing = sum(data_profile.get('missing_values', {}).values())
        if df.shape[0] * df.shape[1] > 0:
            missing_pct = total_missing / (df.shape[0] * df.shape[1]) * 100
        else:
            missing_pct = 0.0
        st.metric("Missing Values", f"{total_missing:,} ({missing_pct:.1f}%)")
    
    # Data preview
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Data profile summary
    with st.expander("Dataset Profile Summary", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Columns:**", len(data_profile['numerical_cols']))
            if data_profile['numerical_cols']:
                st.write(", ".join(data_profile['numerical_cols']))
                
            st.write("**Categorical Columns:**", len(data_profile['categorical_cols']))
            if data_profile['categorical_cols']:
                st.write(", ".join(data_profile['categorical_cols']))
                
        with col2:
            st.write("**Potential Time Series Columns:**", len(data_profile['time_series_candidates']))
            if data_profile['time_series_candidates']:
                st.write(", ".join(data_profile['time_series_candidates']))
                
            st.write("**Columns with Outliers:**", len(data_profile['has_outliers']))
            if data_profile['has_outliers']:
                outlier_items = [f"{col} ({pct:.1f}%)" for col, pct in data_profile['has_outliers'].items()]
                st.write(", ".join(outlier_items))

    # Quality report section
    with st.expander("Data Quality Report", expanded=False):
        score = quality_report.overall_quality_score
        score_color = "🟢" if score >= 0.8 else ("🟡" if score >= 0.5 else "🔴")
        st.metric("Overall Quality Score", f"{score_color} {score:.2f}", help="Composite 0-1 score based on completeness, uniqueness, duplicates, sparse/constant/mixed columns")

        c1, c2, c3 = st.columns(3)
        c1.metric("Completeness", f"{quality_report.completeness:.1%}")
        c2.metric("Uniqueness", f"{quality_report.uniqueness:.1%}")
        c3.metric("Duplicate Rows", f"{quality_report.duplicate_rows:,}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Memory", f"{quality_report.memory_usage_mb:.1f} MB")
        c5.metric("Sparse Columns", len(quality_report.sparse_columns))
        c6.metric("Constant Columns", len(quality_report.constant_columns))

        if quality_report.mixed_type_columns:
            st.write("**Mixed-Type Columns:**", ", ".join(quality_report.mixed_type_columns))

        if quality_report.duplicate_columns_renamed:
            st.write("**Renamed Duplicates:**")
            for old, new in quality_report.duplicate_columns_renamed:
                st.write(f"  `{old}` → `{new}`")

        if quality_report.rows_removed_fully_empty > 0 or quality_report.cols_removed_fully_empty > 0:
            st.write("**Removed:** "
                     f"{quality_report.rows_removed_fully_empty} empty row(s), "
                     f"{quality_report.cols_removed_fully_empty} empty column(s)")

        if score < 0.5:
            st.warning("Low data quality — some analyses or visualizations may be affected.")
    
    # Display recommendations
    st.subheader("Recommended Analyses")
    
    if not recommendations:
        st.info("No applicable analyses found for this dataset. Upload a different file.")
        st.stop()
    
    # Expert mode: raw data view
    if st.session_state.get('_expert_mode'):
        with st.expander("Raw DataFrame", expanded=False):
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="clean_data.csv", mime="text/csv")
        with st.expander("Full Recommendation JSON", expanded=False):
            st.json(recommendations)
    
    for i, rec in enumerate(recommendations[:5]):  # Show top 5 recommendations
        with st.expander(f"📋 {rec['title']} (Relevance: {rec['score']:.2f})", expanded=i==0):
            st.write(rec['description'])
            ci_low = rec.get('score_ci_lower')
            ci_high = rec.get('score_ci_upper')
            if ci_low and ci_high and ci_low != ci_high:
                width = ci_high - ci_low
                label = 'stable' if width < 0.05 else 'moderate' if width < 0.15 else 'uncertain'
                st.caption(f"Confidence interval: [{ci_low:.3f}, {ci_high:.3f}] ({label})")
            
            # Column interestingness badges
            if rec['columns']:
                st.caption("Columns (sorted by interestingness): " + ", ".join(rec['columns']))
            
            # Comparative explanation with #2
            if i == 0 and len(recommendations) > 1:
                show_compare = st.checkbox("Compare with #2", key=f"compare_{rec['type']}_{i}")
                if show_compare:
                    st.markdown(compare_recommendations(rec, recommendations[1]))
            
            show_explanation = st.checkbox(
                "Show explanation",
                key=f"explain_{rec['type']}_{i}"
            )

            if show_explanation:
                st.markdown("---")
                exp = explain_recommendation(rec, data_profile, st.session_state.user_preferences, quality_report)
                for reason in exp['reasons']:
                    st.write(f"• {reason}")
                if exp['technique_reasons']:
                    st.write("**Charts used:**")
                    for t in exp['technique_reasons']:
                        st.write(f"• {t}")
                
                # Counterfactual slider
                st.markdown("---")
                st.caption("**What if?** Adjust this type's priority below to see how the ranking would change:")
                cf_value = st.slider(
                    f"Priority for {rec['type'].replace('_', ' ').title()}",
                    min_value=0.1, max_value=1.0, value=float(rec.get('pref_score', 0.5)),
                    step=0.1, key=f"cf_{rec['type']}_{i}"
                )
                if cf_value != rec.get('pref_score', 0.5):
                    old_pref = rec.get('pref_score', 0.5)
                    scale = cf_value / old_pref if old_pref > 0 else 1.0
                    st.write("**Would-be ranking:**")
                    for j, cr in enumerate(recommendations[:5]):
                        if cr['type'] == rec['type']:
                            new_score = cr['score'] * scale
                        else:
                            new_score = cr['score']
                        arrow = "⬆" if new_score > cr.get('score', 0) else "⬇" if new_score < cr.get('score', 0) else ""
                        st.write(f"{j+1}. {cr['title']}: {new_score:.2f} {arrow}")
                
                st.markdown("---")
                st.markdown("---")
                preference_learner.track_interaction(rec, 'explored', pd.Timestamp.now(), {"event": "show_explanation"})
                st.session_state.user_preferences = preference_learner.preference_weights
            
            # Expert mode: show full recommendation detail
            if st.session_state.get('_expert_mode'):
                st.caption("**Technical Details:**")
                st.json({
                    'type': rec['type'],
                    'base_score': rec.get('base_score'),
                    'pref_score': rec.get('pref_score'),
                    'data_relevance': rec.get('data_relevance'),
                    'quality_adjustment': rec.get('quality_adjustment'),
                    'diversity_penalty': rec.get('diversity_penalty'),
                    'data_factors': rec.get('data_factors'),
                    'columns': rec.get('columns'),
                    'final_score': rec.get('score'),
                    'score_ci': [rec.get('score_ci_lower'), rec.get('score_ci_upper')]
                })
                ci_low = rec.get('score_ci_lower')
                ci_high = rec.get('score_ci_upper')
                if ci_low and ci_high and ci_low != ci_high:
                    st.caption(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}] — {'stable' if ci_high - ci_low < 0.05 else 'moderate' if ci_high - ci_low < 0.15 else 'uncertain'}")
            
            # Visualization based on recommendation type
            selected_cols = []
            if rec['columns']:
                column_options = rec['columns']
                
                # Set default selection based on recommendation type
                if rec['type'] == 'correlation':
                    default_selection = column_options[:min(5, len(column_options))]
                elif rec['type'] == 'categorical':
                    cardinality = st.session_state.get('_cardinality', {})
                    card_subset = {col: cardinality.get(col, df[col].nunique()) for col in column_options}
                    sorted_cols = sorted(card_subset.items(), key=lambda x: x[1])
                    default_selection = [col for col, _ in sorted_cols[:min(3, len(sorted_cols))]]
                else:
                    default_selection = column_options[:min(2, len(column_options))]
                    
                selected_cols = st.multiselect(
                    f"Select columns for {rec['type']} analysis:", 
                    column_options, 
                    default=default_selection
                )
                
                if not selected_cols:
                    st.caption("Select columns above to generate a visualization.")

                if selected_cols:
                    # Update column affinity (co-occurrence tracking)
                    for col in selected_cols:
                        st.session_state._col_frequency[col] = st.session_state._col_frequency.get(col, 0) + 1
                    for i_col in selected_cols:
                        for j_col in selected_cols:
                            if i_col < j_col:
                                if i_col not in st.session_state._col_cooccurrence:
                                    st.session_state._col_cooccurrence[i_col] = {}
                                if j_col not in st.session_state._col_cooccurrence:
                                    st.session_state._col_cooccurrence[j_col] = {}
                                st.session_state._col_cooccurrence[i_col][j_col] = st.session_state._col_cooccurrence[i_col].get(j_col, 0) + 1
                                st.session_state._col_cooccurrence[j_col][i_col] = st.session_state._col_cooccurrence[j_col].get(i_col, 0) + 1

                    # Add visualization type selection
                    visualization_options = {}
                    if rec['type'] == 'correlation':
                        visualization_options = {
                            'side_by_side': 'Correlation Matrix (Complete View)'
                        }
                    elif rec['type'] == 'categorical':
                        visualization_options = {
                            'bar': 'Bar Chart',
                            'pie': 'Pie Chart'
                        }
                    
                    # Only show visualization type selector if we have options for this visualization type
                    selected_viz_type = None
                    if visualization_options:
                        selected_viz_type = st.selectbox(
                            f"Select visualization type:",
                            options=list(visualization_options.keys()),
                            format_func=lambda x: visualization_options[x]
                        )
                    
                    # Generate and show visualization
                    viz_placeholder = st.empty()
                    viz_placeholder.caption("Rendering chart...")
                    try:
                        kwargs = {}
                        if selected_viz_type:
                            kwargs['plot_type'] = selected_viz_type
                            
                        fig = visualization_generator.generate_visualization(
                            rec['type'], df, selected_cols, quality_report=quality_report, **kwargs
                        )
                        st.session_state._viz_start_time[rec['type']] = time.time()
                        event = viz_placeholder.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"viz_{rec['type']}_{i}")
                        if event and event.selection and event.selection.points:
                            st.caption(f"Selected: {event.selection.points[0]}")
                        preference_learner.track_interaction(rec, 'column_selected', pd.Timestamp.now(), {"columns": selected_cols, "viz_type": selected_viz_type})
                        st.session_state._viewed_combos.add((rec['type'], frozenset(selected_cols)))
                        st.session_state.user_preferences = preference_learner.preference_weights
                        
                        # Per-row outlier explanation for outlier analysis
                        if rec['type'] == 'outliers' and selected_cols:
                            outlier_rows = pd.DataFrame()
                            for col in selected_cols:
                                if df[col].dtype.kind not in 'ifc':
                                    continue
                                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                                iqr = q3 - q1
                                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                                mask = (df[col] < lower) | (df[col] > upper)
                                flagged = df[mask].copy()
                                if not flagged.empty:
                                    flagged['_outlier_col'] = col
                                    flagged['_direction'] = flagged[col].apply(
                                        lambda v: f"+{v - upper:.2f}" if v > upper else f"{v - lower:.2f}"
                                    )
                                    outlier_rows = pd.concat([outlier_rows, flagged])
                            if not outlier_rows.empty:
                                if st.checkbox("Show outlier rows", key=f"_outlier_rows_{i}"):
                                    st.dataframe(
                                        outlier_rows.drop_duplicates().reset_index(drop=True),
                                        use_container_width=True,
                                        column_order=[c for c in outlier_rows.columns if not c.startswith('_')] + ['_outlier_col', '_direction']
                                    )
                    except Exception as e:
                        viz_placeholder.error(f"Error generating visualization: {str(e)}")
            
            if st.session_state.get('ai_enabled'):
                st.markdown("**AI Analysis**")
                if not selected_cols:
                    st.caption("Select columns above to generate AI analysis.")
                else:
                    container = st.container()
                    with container:
                        provider = st.session_state.get('_ai_provider', 'local')
                        model = st.session_state.get('_llm_model', llm_adapter.DEFAULT_MODEL)
                        endpoint = st.session_state.get('_llm_endpoint')

                        if provider != "local" and not endpoint:
                            st.caption("Configure an API key and provider in the sidebar.")
                        else:
                            ai_key = f"_ai_{provider}_{rec['type']}_{i}_{hash(tuple(selected_cols))}"
                            if ai_key not in st.session_state:
                                st.session_state[ai_key] = None
                            if st.session_state[ai_key] is None:
                                with st.spinner(f"Querying {model}..."):
                                    if provider == "local":
                                        result = llm_adapter.generate_analysis_local(
                                            data_profile, df, rec['type'], selected_cols, model=model
                                        )
                                    else:
                                        key_map = {"openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY", "custom": "CUSTOM_API_KEY"}
                                        api_key = os.environ.get(key_map.get(provider, ""), "")
                                        result = llm_adapter.generate_analysis_remote(
                                            data_profile, df, rec['type'], selected_cols,
                                            api_key=api_key, endpoint=endpoint, model=model
                                        )
                                    st.session_state[ai_key] = (
                                        result["text"] if result.get("ok")
                                        else f"_error:{result.get('error')}"
                                    )
                            cached = st.session_state[ai_key]
                            if cached.startswith("_error:"):
                                st.caption(f"LLM unavailable — {cached[7:]}")
                            else:
                                st.markdown(cached)
            
            # User feedback
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button(f"👍 Useful — {rec['title']}", key=f"useful_{i}"):
                    dwell = time.time() - st.session_state._viz_start_time.get(rec['type'], time.time())
                    preference_learner.track_interaction(rec, 'liked', pd.Timestamp.now(), {"dwell_seconds": round(dwell, 1), "abandoned": dwell < 5})
                    st.session_state.user_preferences = preference_learner.preference_weights
                    st.session_state.recommendations = recommendation_engine.generate_recommendations(
                        data_profile,
                        st.session_state.user_preferences,
                        quality_score=st.session_state.quality_report.overall_quality_score,
                        viewed_combos=st.session_state._viewed_combos,
                        interaction_history=st.session_state.interaction_history,
                        col_affinity=_compute_col_affinity()
                    )
                    st.toast("Thanks! Prioritizing similar analyses.", icon="👍")
                    
            with col2:
                if st.button(f"👎 Not Useful — {rec['title']}", key=f"not_useful_{i}"):
                    dwell = time.time() - st.session_state._viz_start_time.get(rec['type'], time.time())
                    preference_learner.track_interaction(rec, 'disliked', pd.Timestamp.now(), {"dwell_seconds": round(dwell, 1), "abandoned": dwell < 5})
                    st.session_state.user_preferences = preference_learner.preference_weights
                    st.session_state.recommendations = recommendation_engine.generate_recommendations(
                        data_profile,
                        st.session_state.user_preferences,
                        quality_score=st.session_state.quality_report.overall_quality_score,
                        viewed_combos=st.session_state._viewed_combos,
                        interaction_history=st.session_state.interaction_history,
                        col_affinity=_compute_col_affinity()
                    )
                    st.toast("Noted. Showing fewer analyses like this.", icon="👎")
            
            with col3:
                if st.button(f"⏭️ Skip — {rec['title']}", key=f"skip_{i}"):
                    dwell = time.time() - st.session_state._viz_start_time.get(rec['type'], time.time())
                    preference_learner.track_interaction(rec, 'ignored', pd.Timestamp.now(), {"dwell_seconds": round(dwell, 1), "abandoned": dwell < 5})
                    st.session_state.user_preferences = preference_learner.preference_weights
                    st.toast("Skipped — fewer similar suggestions if repeated.", icon="⏭️")
    
else:
    # Display welcome message and instructions when no file is uploaded
    st.markdown("""
    ## Welcome to the EDA Assistant!
    
    Upload a dataset to get analysis recommendations, generate charts, and explore your data.
    
    ### How it works:
    
    1. **Upload a dataset** using the sidebar
    2. **Browse recommendations** ranked by data characteristics and your priorities
    3. **Give feedback** (👍/👎) to shift future recommendations
    4. **Generate visualizations** with column selection
    
    ### Get started:
    
    Upload a CSV, Excel or JSON file using the file uploader in the sidebar.
    """)
    
    st.info(
        "Supported formats: **CSV**, **XLSX**, **XLS**, **JSON** — "
        "files up to ~50 MB recommended."
    )

# Footer
st.markdown("---")
st.markdown("EDA Assistant")

 