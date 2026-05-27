import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys, os


def _load_env(path=".env"):
    """Minimal .env loader — no external dependency needed."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())
    except FileNotFoundError:
        pass


_load_env()

# Import custom modules
from data_processor import DataProcessor
from recommendation_engine import RecommendationEngine
from preference_learner import PreferenceTracker
from insight_generator import explain_recommendation, explain_user_preferences
from visualization_generator import VisualizationGenerator
from constants import DEFAULT_PREFERENCES
from data_quality import QualityReport
import llm_adapter

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
    if st.session_state.user_preferences:
        prefs = st.session_state.user_preferences
        pref_df = pd.DataFrame([
            {'type': k.replace('_', ' ').title(), 'score': v, 'default': v == 0.5}
            for k, v in prefs.items()
        ]).sort_values('score', ascending=True)

        fig = px.bar(
            pref_df, x='score', y='type', orientation='h',
            text='score', range_x=[0, 1.1],
            color='default', color_discrete_map={True: '#e0e0e0', False: '#1f77b4'},
            height=220
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(
            showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
            xaxis_visible=False, yaxis_title=None
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown("#### Current Priority Levels")
        st.write(explain_user_preferences(prefs))
        
        st.markdown("#### Set Priorities")
        st.write("Adjust how much each analysis type is weighted in recommendations:")
        
        # Create a form for preference adjustments
        with st.form(key="preference_form"):
            new_preferences = {}
            for analysis_type, current_value in st.session_state.user_preferences.items():
                # Create a more user-friendly label
                friendly_name = analysis_type.replace('_', ' ').title()
                new_preferences[analysis_type] = st.slider(
                    f"{friendly_name}",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(current_value),
                    step=0.1,
                    format="%.1f"
                )
            
            # Submit button for the form
            submit_button = st.form_submit_button(label="Update Preferences")
            
            if submit_button:
                st.session_state.user_preferences = preference_learner.set_preferences(new_preferences)

                if st.session_state.data_profile is not None:
                    st.session_state.recommendations = recommendation_engine.generate_recommendations(
                        st.session_state.data_profile,
                        st.session_state.user_preferences
                    )

                st.toast("Preferences updated — recommendations reordered.", icon="✅")
    
    # Reset preferences button
    if st.button("Reset Preferences"):
        st.session_state.user_preferences = DEFAULT_PREFERENCES.copy()
        preference_learner.preference_weights = st.session_state.user_preferences
        st.session_state.interaction_history = []
        st.toast("Preferences reset to defaults.", icon="🔄")
    
    st.header("AI Analysis")
    has_api_key = any(os.environ.get(k) for k in ["OPENROUTER_API_KEY", "GROQ_API_KEY"])
    default_ai_enabled = has_api_key or bool(os.environ.get("LLM_PROVIDER"))
    st.session_state.ai_enabled = st.toggle(
        "Enable AI insights",
        value=st.session_state.get('ai_enabled', default_ai_enabled),
        help="Use a local model (Ollama) or a free remote API"
    )
    if st.session_state.ai_enabled:
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
            progress.progress(35, text="Profiling dataset...")
            data_profile = data_processor.profile_dataset(df)
            st.session_state.data_profile = data_profile
            
            # Cache column cardinalities for the recommendation loop
            st.session_state._cardinality = {col: df[col].nunique() for col in df.columns}
            
            progress.progress(70, text="Generating recommendations...")
            recommendations = recommendation_engine.generate_recommendations(
                data_profile, 
                st.session_state.user_preferences
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
    if unnamed_cols:
        st.subheader("Unnamed Columns Detected")
        st.write(f"Found {len(unnamed_cols)} column(s) without names.")
        if st.session_state.get('ai_enabled') and '_column_names' not in st.session_state:
            with st.spinner("Suggesting column names from data..."):
                provider = st.session_state.get('_ai_provider', 'local')
                model = st.session_state.get('_llm_model', llm_adapter.DEFAULT_MODEL)
                endpoint = st.session_state.get('_llm_endpoint')
                key_map = {"openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY", "custom": "CUSTOM_API_KEY"}
                api_key = os.environ.get(key_map.get(provider, ""), "")
                result = llm_adapter.suggest_column_names(
                    df, unnamed_cols, model=model, provider=provider,
                    api_key=api_key, endpoint=endpoint
                )
                if result.get("ok"):
                    st.session_state._column_names = result["names"]
                else:
                    st.info(f"Could not generate suggestions: {result.get('error')}")
                    st.session_state._column_names = {}

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
                st.session_state.data_profile = data_processor.profile_dataset(df)
                st.session_state._cardinality = {col: df[col].nunique() for col in df.columns}
                st.session_state.recommendations = recommendation_engine.generate_recommendations(
                    st.session_state.data_profile, st.session_state.user_preferences
                )
                st.rerun()
    
    # ── Quality warnings ─────────────────────────────────────
    if quality_report.warnings:
        for w in quality_report.warnings[:5]:
            st.warning(w)
        if len(quality_report.warnings) > 5:
            with st.expander(f"+{len(quality_report.warnings) - 5} more warnings"):
                for w in quality_report.warnings[5:]:
                    st.warning(w)

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
    
    for i, rec in enumerate(recommendations[:5]):  # Show top 5 recommendations
        with st.expander(f"📋 {rec['title']} (Relevance: {rec['score']:.2f})", expanded=i==0):
            st.write(rec['description'])
            
            show_explanation = st.checkbox(
                "Show explanation",
                key=f"explain_{rec['type']}_{i}"
            )

            if show_explanation:
                st.markdown("---")
                exp = explain_recommendation(rec, data_profile, st.session_state.user_preferences)
                for reason in exp['reasons']:
                    st.write(f"• {reason}")
                if exp['technique_reasons']:
                    st.write("**Charts used:**")
                    for t in exp['technique_reasons']:
                        st.write(f"• {t}")
                st.markdown("---")
            
            # Visualization based on recommendation type
            selected_cols = []
            if rec['columns']:
                column_options = rec['columns']
                
                # Set default selection based on recommendation type
                if rec['type'] == 'correlation':
                    # For correlation, select all numerical columns by default (up to 5)
                    default_selection = column_options[:min(5, len(column_options))]
                elif rec['type'] == 'categorical':
                    # For categorical, start with low cardinality columns
                    cardinality = st.session_state.get('_cardinality', {})
                    card_subset = {col: cardinality.get(col, df[col].nunique()) for col in column_options}
                    sorted_cols = sorted(card_subset.items(), key=lambda x: x[1])
                    default_selection = [col for col, _ in sorted_cols[:min(3, len(sorted_cols))]]
                else:
                    # For other types, select first 2 columns by default
                    default_selection = column_options[:min(2, len(column_options))]
                    
                selected_cols = st.multiselect(
                    f"Select columns for {rec['type']} analysis:", 
                    column_options, 
                    default=default_selection
                )
                
                if not selected_cols:
                    st.caption("Select columns above to generate a visualization.")
                
                if selected_cols:
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
                            rec['type'], df, selected_cols, **kwargs
                        )
                        viz_placeholder.plotly_chart(fig, use_container_width=True)
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
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(f"👍 Useful — {rec['title']}", key=f"useful_{i}"):
                    preference_learner.track_interaction(rec, 'liked', pd.Timestamp.now())
                    st.session_state.user_preferences = preference_learner.preference_weights
                    st.session_state.recommendations = recommendation_engine.generate_recommendations(
                        data_profile,
                        st.session_state.user_preferences
                    )
                    st.toast("Thanks! Prioritizing similar analyses.", icon="👍")
                    
            with col2:
                if st.button(f"👎 Not Useful — {rec['title']}", key=f"not_useful_{i}"):
                    preference_learner.track_interaction(rec, 'disliked', pd.Timestamp.now())
                    st.session_state.user_preferences = preference_learner.preference_weights
                    st.session_state.recommendations = recommendation_engine.generate_recommendations(
                        data_profile,
                        st.session_state.user_preferences
                    )
                    st.toast("Noted. Showing fewer analyses like this.", icon="👎")
    
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

 