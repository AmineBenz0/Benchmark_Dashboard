import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

# Set page config (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="LLM Metrics Dashboard", layout="wide")

# Load the aggregated data from a folder
def normalize_metric(value, min_value, max_value):
    """Normalize a metric to the range [0, 1]."""
    return (value - min_value) / (max_value - min_value)

@st.cache_data
def load_data_from_folder(folder_path):
    metrics_data = []
    queries_data = []
    # Define expected ranges for normalization
    correctness_range = (0.0, 1.0)        # Correctness is already between 0 and 1
    relevance_range = (0.0, 1.0)          # Relevance is already between 0 and 1
    perplexity_range = (1.0, 10.0)        # Example range for perplexity
    resolution_time_range = (0.5, 5.0)    # Example range for resolution time

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            model_name = filename.replace("_aggregated_results.json", "")
            with open(os.path.join(folder_path, filename), "r") as f:
                model_data = json.load(f)
                # Normalize metrics
                correctness = normalize_metric(model_data["correctness"], *correctness_range)
                relevance = normalize_metric(model_data["relevance"], *relevance_range)
                inverted_perplexity = 1 / model_data["perplexity"]
                perplexity = normalize_metric(inverted_perplexity, 1 / perplexity_range[1], 1 / perplexity_range[0])
                resolution_time = normalize_metric(model_data["resolution_time"], *resolution_time_range)
                # Weights for combined_score
                weights = {
                    "correctness": 0.4,
                    "relevance": 0.3,
                    "perplexity": 0.2,
                    "resolution_time": 0.1,
                }
                # Calculate combined_score (ensures it stays within [0, 1])
                combined_score = (
                    correctness * weights["correctness"] +
                    relevance * weights["relevance"] +
                    perplexity * weights["perplexity"] -
                    resolution_time * weights["resolution_time"]
                )
                # Ensure combined_score doesn't drop below 0
                combined_score = max(0, combined_score)
                # Extract metrics
                metrics = {
                    "model": model_name,
                    "correctness": model_data["correctness"],
                    "relevance": model_data["relevance"],
                    "perplexity": model_data["perplexity"],
                    "resolution_time": model_data["resolution_time"],
                    "combined_score": combined_score,
                    "cpu_%": model_data["cpu_%"],
                    "ram_%": model_data["ram_%"],
                    "gpu_%": model_data["gpu_%"],
                    "gpu_mem_%": model_data["gpu_mem_%"],
                }
                metrics_data.append(metrics)
                # Extract queries
                for query in model_data["queries"]:
                    query["model"] = model_name
                    queries_data.append(query)
    return pd.DataFrame(metrics_data), pd.DataFrame(queries_data)

# Render the dashboard for a specific dataframe
def render_dashboard(metrics_df, queries_df, title):
    st.title(title)
    
    # Extract unique models
    unique_models = metrics_df["model"].unique()

    # Add a sidebar for filters
    st.sidebar.header("Filters")
    selected_models = st.sidebar.multiselect("Select models", unique_models, default=unique_models)

    # Filter data based on selected models
    filtered_metrics = metrics_df[metrics_df["model"].isin(selected_models)]
    filtered_queries = queries_df[queries_df["model"].isin(selected_models)]

    # ========================== PERFORMANCE METRICS ==========================
    st.header("Performance Metrics")

    # Correctness (Higher is better)
    st.subheader("Correctness (Higher is better)")
    fig1 = px.bar(filtered_metrics, x="model", y="correctness", title="Correctness Comparison")
    st.plotly_chart(fig1, use_container_width=True)

    # Relevance (Higher is better)
    st.subheader("Relevance (Higher is better)")
    fig2 = px.bar(filtered_metrics, x="model", y="relevance", title="Relevance Comparison")
    st.plotly_chart(fig2, use_container_width=True)

    # Combined Score (Higher is better)
    st.subheader("Combined Score (Higher is better)")
    fig3 = px.bar(filtered_metrics, x="model", y="combined_score", title="Combined Score Comparison")
    st.plotly_chart(fig3, use_container_width=True)

    # ========================== EFFICIENCY METRICS ==========================
    st.header("Efficiency Metrics")

    # Resolution Time (Lower is better)
    st.subheader("Resolution Time (Lower is better)")
    fig4 = px.bar(filtered_metrics, x="model", y="resolution_time", title="Resolution Time Comparison")
    st.plotly_chart(fig4, use_container_width=True)

    # Perplexity (Lower is better)
    st.subheader("Perplexity (Lower is better)")
    fig5 = px.bar(filtered_metrics, x="model", y="perplexity", title="Perplexity Comparison")
    st.plotly_chart(fig5, use_container_width=True)

    # ========================== RESOURCE USAGE METRICS ==========================
    st.header("Resource Usage Metrics")
    resource_metrics = ["cpu_%", "ram_%", "gpu_mem_%"]
    fig6 = px.bar(
        filtered_metrics.melt(id_vars=["model"], value_vars=resource_metrics, var_name="Metric", value_name="Percentage"),
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
    model_queries = filtered_queries[filtered_queries["model"] == selected_model][["query", "context", "final_answer"]]

    # Display Table with Query, Context, and Response
    st.subheader(f"Responses for Model: {selected_model}")
    st.dataframe(model_queries)

    # ========================== SUMMARY TABLE ==========================
    st.header("Summary Table")
    st.table(filtered_metrics)

# Main function
def main():
    # Add a sidebar for selecting the LLM type
    st.sidebar.title("LLM Type")
    llm_type = st.sidebar.radio("Select LLM Type", ["Small LLMs", "Big LLMs"])

    # Load data based on the selected LLM type
    if llm_type == "Small LLMs":
        metrics_df, queries_df = load_data_from_folder("Small_aggregated/")
        render_dashboard(metrics_df, queries_df, "Small LLM Metrics Dashboard")
    else:
        metrics_df, queries_df = load_data_from_folder("Big_aggregated/")
        render_dashboard(metrics_df, queries_df, "Big LLM Metrics Dashboard")

if __name__ == "__main__":
    main()
