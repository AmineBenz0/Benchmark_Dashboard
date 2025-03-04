import streamlit as st
import pandas as pd
import plotly.express as px
import json 

# Set page config (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="Model Metrics Dashboard", layout="wide")

# Load the aggregated data
@st.cache_data
def load_data():
    with open("aggregated_metrics.json", "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

df = load_data()

# Extract unique models
unique_models = df["model"].unique()

# Exclude non-numeric columns before aggregation
numeric_columns = ["correctness", "relevance", "rouge", "bleu", "perplexity", "resolution_time",
                   "combined_score", "cpu_%", "ram_%", "gpu_%", "gpu_mem_%"]

# Compute mean metrics per model
aggregated_metrics = df.groupby("model")[numeric_columns].mean().reset_index()

# Page title
st.title("Model Metrics Dashboard")

# Add a sidebar for filters
st.sidebar.header("Filters")
selected_models = st.sidebar.multiselect("Select models", unique_models, default=unique_models)

# Filter data based on selected models
filtered_df = aggregated_metrics[aggregated_metrics["model"].isin(selected_models)]

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

# ROUGE (Higher is better)
st.subheader("ROUGE (Higher is better)")
fig3 = px.bar(filtered_df, x="model", y="rouge", title="ROUGE Comparison")
st.plotly_chart(fig3, use_container_width=True)

# BLEU (Higher is better)
st.subheader("BLEU (Higher is better)")
fig4 = px.bar(filtered_df, x="model", y="bleu", title="BLEU Comparison")
st.plotly_chart(fig4, use_container_width=True)

# Combined Score (Higher is better)
st.subheader("Combined Score (Higher is better)")
fig5 = px.bar(filtered_df, x="model", y="combined_score", title="Combined Score Comparison")
st.plotly_chart(fig5, use_container_width=True)

# ========================== EFFICIENCY METRICS ==========================
st.header("Efficiency Metrics")

# Resolution Time (Lower is better)
st.subheader("Resolution Time (Lower is better)")
fig6 = px.bar(filtered_df, x="model", y="resolution_time", title="Resolution Time Comparison")
st.plotly_chart(fig6, use_container_width=True)

# Perplexity (Lower is better)
st.subheader("Perplexity (Lower is better)")
fig7 = px.bar(filtered_df, x="model", y="perplexity", title="Perplexity Comparison")
st.plotly_chart(fig7, use_container_width=True)

# ========================== RESOURCE USAGE METRICS ==========================
st.header("Resource Usage Metrics")
resource_metrics = ["cpu_%", "ram_%", "gpu_mem_%"]
fig8 = px.bar(
    filtered_df.melt(id_vars=["model"], value_vars=resource_metrics, var_name="Metric", value_name="Percentage"),
    x="model",
    y="Percentage",
    color="Metric",
    barmode="group",
    title="Resource Usage Comparison"
)
st.plotly_chart(fig8, use_container_width=True)

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
