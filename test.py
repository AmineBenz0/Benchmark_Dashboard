import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

# Set page config (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="LLM Metrics Dashboard", layout="wide")

# Load the aggregated data from a folder
@st.cache_data
def load_data_from_folder(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            model_name = filename.replace("_aggregated_results.json", "")
            with open(os.path.join(folder_path, filename), "r") as f:
                model_data = json.load(f)
                # Add model name to the data
                model_data["model"] = model_name
                all_data.append(model_data)
    return pd.DataFrame(all_data)

# Extract metrics for a given dataframe
def extract_metrics(df):
    # Exclude non-numeric columns before aggregation
    numeric_columns = ["correctness", "relevance", "perplexity", "resolution_time",
                       "combined_score", "cpu_%", "ram_%", "gpu_%", "gpu_mem_%"]
    return df.groupby("model")[numeric_columns].mean().reset_index()

# Render the dashboard for a specific dataframe
def render_dashboard(df, title):
    st.title(title)
    
    # Extract unique models
    unique_models = df["model"].unique()

    # Add a sidebar for filters
    st.sidebar.header("Filters")
    selected_models = st.sidebar.multiselect("Select models", unique_models, default=unique_models)

    # Filter data based on selected models
    filtered_df = df[df["model"].isin(selected_models)]

    # ========================== PERFORMANCE METRICS ==========================
    st.header("Performance Metrics")

    # Correctness (Higher is better)
    st.subheader("Correctness (Higher is better)")
    fig1 = px.bar(filtered_df, x="model", y="correctness", title="Correctness Comparison")
    st.plotly_chart(fig1, use_container_width=True)

    # Relevance (Higher is better)
    st.subheader("Relevance (Higher is better)")
    fig2 = px.bar(filtered_df, x="model", y="relevance", title="Relevance Comparison")
    st.plotly_chart(fig2, use_container_width=True)

    # Combined Score (Higher is better)
    st.subheader("Combined Score (Higher is better)")
    fig3 = px.bar(filtered_df, x="model", y="combined_score", title="Combined Score Comparison")
    st.plotly_chart(fig3, use_container_width=True)

    # ========================== EFFICIENCY METRICS ==========================
    st.header("Efficiency Metrics")

    # Resolution Time (Lower is better)
    st.subheader("Resolution Time (Lower is better)")
    fig4 = px.bar(filtered_df, x="model", y="resolution_time", title="Resolution Time Comparison")
    st.plotly_chart(fig4, use_container_width=True)

    # Perplexity (Lower is better)
    st.subheader("Perplexity (Lower is better)")
    fig5 = px.bar(filtered_df, x="model", y="perplexity", title="Perplexity Comparison")
    st.plotly_chart(fig5, use_container_width=True)

    # ========================== RESOURCE USAGE METRICS ==========================
    st.header("Resource Usage Metrics")
    resource_metrics = ["cpu_%", "ram_%", "gpu_mem_%"]
    fig6 = px.bar(
        filtered_df.melt(id_vars=["model"], value_vars=resource_metrics, var_name="Metric", value_name="Percentage"),
        x="model",
        y="Percentage",
        color="Metric",
        barmode="group",
        title="Resource Usage Comparison"
    )
    st.plotly_chart(fig6, use_container_width=True)

    # ========================== SHOW INPUT, CONTEXT, AND OUTPUT ==========================
    st.header("📜 Model Responses Showcase")

    # Dropdown to select a single model for detailed response view
    selected_model = st.selectbox("Select a model to view responses:", unique_models)

    # Extract relevant responses for the selected model
    model_queries = df[df["model"] == selected_model][["query", "context", "final_answer"]]

    # Display Table with Query, Context, and Response
    st.subheader(f"Responses for Model: {selected_model}")
    st.dataframe(model_queries)

    # ========================== SUMMARY TABLE ==========================
    st.header("Summary Table")
    st.table(filtered_df)

# Main function
def main():
    # Add a sidebar for selecting the LLM type
    st.sidebar.title("LLM Type")
    llm_type = st.sidebar.radio("Select LLM Type", ["Small LLMs", "Big LLMs"])

    # Load data based on the selected LLM type
    if llm_type == "Small LLMs":
        df = load_data_from_folder("Small_Aggregated/")
        render_dashboard(df, "Small LLM Metrics Dashboard")
    else:
        df = load_data_from_folder("Big_aggregated/")
        render_dashboard(df, "Big LLM Metrics Dashboard")

if __name__ == "__main__":
    main()
