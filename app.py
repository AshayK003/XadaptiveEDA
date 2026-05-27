import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
from data_processor import DataProcessor
from recommendation_engine import RecommendationEngine
from preference_learner import PreferenceTracker
from insight_generator import explain_recommendation, explain_user_preferences, generate_insights
from visualization_generator import VisualizationGenerator
from constants import DEFAULT_PREFERENCES

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
    .st-emotion-cache-nahz7x {
        font-size: 0.8rem;
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
    
    st.header("Your Priorities")
    if st.session_state.user_preferences:
        preferences_df = pd.DataFrame({
            'Analysis Type': list(st.session_state.user_preferences.keys()),
            'Preference Score': list(st.session_state.user_preferences.values())
        })
        
        preferences_df = preferences_df.sort_values('Preference Score', ascending=False)
        st.bar_chart(preferences_df.set_index('Analysis Type'))
        
        st.markdown("#### Current Priority Levels")
        st.write(explain_user_preferences(st.session_state.user_preferences))
        
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

# Main content area
if uploaded_file is not None:
    # Load and process data if not already done
    if st.session_state.df is None or uploaded_file.name != getattr(st.session_state, 'last_file', ''):
        try:
            with st.spinner("Processing dataset..."):
                df = data_processor.load_data(uploaded_file)
                st.session_state.df = df
                st.session_state.last_file = uploaded_file.name
                
                # Profile the dataset
                data_profile = data_processor.profile_dataset(df)
                st.session_state.data_profile = data_profile
                
                # Generate initial recommendations
                recommendations = recommendation_engine.generate_recommendations(
                    data_profile, 
                    st.session_state.user_preferences
                )
                st.session_state.recommendations = recommendations
        except Exception as e:
            import sys, os
            diag = f"Error type: {type(e).__name__}\nFile: {uploaded_file.name}\nExtension: {os.path.splitext(uploaded_file.name)[1].lower()}\nPython: {sys.executable}"
            st.error(f"Error processing file: {str(e)}\n\n{diag}")
            st.stop()
    else:
        # Use cached data
        df = st.session_state.df
        data_profile = st.session_state.data_profile
        recommendations = st.session_state.recommendations
    
    # Display data overview
    st.subheader("Data Overview")
    
    # Show shape and sample
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        if data_profile['missing_values'] and df.shape[0] > 0:
            total_missing = sum(data_profile['missing_values'].values())
            missing_pct = total_missing / (df.shape[0] * df.shape[1]) * 100
            st.metric("Missing Values", f"{total_missing:,} ({missing_pct:.1f}%)")
    
    # Data preview
    with st.expander("Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Data profile summary
    with st.expander("Dataset Profile Summary"):
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
    
    # Display recommendations
    st.subheader("Recommended Analyses")
    
    for i, rec in enumerate(recommendations[:5]):  # Show top 5 recommendations
        with st.expander(f"📋 {rec['title']} (Relevance: {rec['score']:.2f})", expanded=i==0):
            st.write(rec['description'])
            
            toggle_key = 'show_why_' + rec['type'] + '_' + str(i)
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = False

            if st.button("📖 Why this recommendation?", key=f"why_btn_{rec['type']}_{i}"):
                st.session_state[toggle_key] = not st.session_state[toggle_key]

            if st.session_state[toggle_key]:
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
            if rec['columns']:
                column_options = rec['columns']
                
                # Set default selection based on recommendation type
                if rec['type'] == 'correlation':
                    # For correlation, select all numerical columns by default (up to 5)
                    default_selection = column_options[:min(5, len(column_options))]
                elif rec['type'] == 'categorical':
                    # For categorical, start with low cardinality columns
                    cardinality = {col: df[col].nunique() for col in column_options}
                    sorted_cols = sorted(cardinality.items(), key=lambda x: x[1])
                    default_selection = [col for col, _ in sorted_cols[:min(3, len(sorted_cols))]]
                else:
                    # For other types, select first 2 columns by default
                    default_selection = column_options[:min(2, len(column_options))]
                    
                selected_cols = st.multiselect(
                    f"Select columns for {rec['type']} analysis:", 
                    column_options, 
                    default=default_selection
                )
                
                if selected_cols:
                    # Track this as a selection
                    preference_learner.track_interaction(rec, 'selected', pd.Timestamp.now())
                    st.session_state.user_preferences = preference_learner.preference_weights
                    
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
                    try:
                        with st.spinner("Generating visualization..."):
                            kwargs = {}
                            if selected_viz_type:
                                kwargs['plot_type'] = selected_viz_type
                                
                            fig = visualization_generator.generate_visualization(
                                rec['type'], df, selected_cols, **kwargs
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating visualization: {str(e)}")
            
            # User feedback
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("👍 Useful", key=f"useful_{i}"):
                    preference_learner.track_interaction(rec, 'liked', pd.Timestamp.now())
                    st.session_state.user_preferences = preference_learner.preference_weights
                    st.toast("Thanks! Prioritizing similar analyses.", icon="👍")
                    
                    # Update recommendations based on new preferences
                    st.session_state.recommendations = recommendation_engine.generate_recommendations(
                        data_profile, 
                        st.session_state.user_preferences
                    )
                    st.rerun()
                    
            with col2:
                if st.button("👎 Not Useful", key=f"not_useful_{i}"):
                    preference_learner.track_interaction(rec, 'disliked', pd.Timestamp.now())
                    st.session_state.user_preferences = preference_learner.preference_weights
                    st.toast("Noted. Showing fewer analyses like this.", icon="👎")
                    
                    # Update recommendations based on new preferences
                    st.session_state.recommendations = recommendation_engine.generate_recommendations(
                        data_profile, 
                        st.session_state.user_preferences
                    )
                    st.rerun()
    
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
    
    # Sample datasets section
    st.subheader("Don't have a dataset? Try one of these examples:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Iris Flower Dataset**
        - 150 samples of iris flowers
        - Features: sepal length/width, petal length/width
        - Classification task
        
        [Download CSV](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
        """)
        
    with col2:
        st.markdown("""
        **Titanic Passengers**
        - 891 passengers from the Titanic
        - Features: age, class, fare, gender
        - Missing values and categorical data
        
        [Download CSV](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
        """)
        
    with col3:
        st.markdown("""
        **Auto MPG Dataset**
        - 398 automobiles 
        - Features: mpg, cylinders, horsepower
        - Regression task with outliers
        
        [Download CSV](https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv)
        """)

# Footer
st.markdown("---")
st.markdown(
    "EDA Assistant • Built with Streamlit • "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
) 