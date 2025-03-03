import streamlit as st
import pandas as pd
import plotly.express as px
import json

# Set page config (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="Model Metrics Dashboard", layout="wide")

# Load the aggregated data
@st.cache_data  # Cache the data for better performance
def load_data():
    with open("aggregated_metrics.json", "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

df = load_data()

# Page title
st.title("Model Metrics Dashboard")

# Add a sidebar for filters
st.sidebar.header("Filters")
selected_models = st.sidebar.multiselect("Select models", df["model"].unique(), default=df["model"].unique())

# Filter the data based on selected models
filtered_df = df[df["model"].isin(selected_models)]

# ========================== PERFORMANCE METRICS ==========================
st.header("Performance Metrics")

# Correctness (Higher is better)
st.subheader("Correctness (Higher is better)")
fig1 = px.bar(
    filtered_df,
    x="model",
    y="correctness",
    title="Correctness Comparison"
)
st.plotly_chart(fig1, use_container_width=True)

# Relevance (Higher is better)
st.subheader("Relevance (Higher is better)")
fig2 = px.bar(
    filtered_df,
    x="model",
    y="relevance",
    title="Relevance Comparison"
)
st.plotly_chart(fig2, use_container_width=True)

# ROUGE (Higher is better)
st.subheader("ROUGE (Higher is better)")
fig3 = px.bar(
    filtered_df,
    x="model",
    y="rouge",
    title="ROUGE Comparison"
)
st.plotly_chart(fig3, use_container_width=True)

# BLEU (Higher is better)
st.subheader("BLEU (Higher is better)")
fig4 = px.bar(
    filtered_df,
    x="model",
    y="bleu",
    title="BLEU Comparison"
)
st.plotly_chart(fig4, use_container_width=True)

# Combined Score (Higher is better)
st.subheader("Combined Score (Higher is better)")
fig5 = px.bar(
    filtered_df,
    x="model",
    y="combined_score",
    title="Combined Score Comparison"
)
st.plotly_chart(fig5, use_container_width=True)

# ========================== EFFICIENCY METRICS ==========================
st.header("Efficiency Metrics")

# Resolution Time (Lower is better)
st.subheader("Resolution Time (Lower is better)")
fig6 = px.bar(
    filtered_df,
    x="model",
    y="resolution_time",
    title="Resolution Time Comparison",
    labels={"resolution_time": "Resolution Time (s)"}
)
st.plotly_chart(fig6, use_container_width=True)

# Perplexity (Lower is better)
st.subheader("Perplexity (Lower is better)")
fig7 = px.bar(
    filtered_df,
    x="model",
    y="perplexity",
    title="Perplexity Comparison",
    labels={"perplexity": "Perplexity"}
)
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
selected_model = st.selectbox("Select a model to view responses:", df["model"].unique())

# Extract the relevant data
with open("aggregated_metrics.json", "r") as f:
    data = json.load(f)

# Find the responses of the selected model
model_data = next((entry for entry in data if entry["model"] == selected_model), None)

if model_data and "queries" in model_data:
    # Convert query data to a DataFrame
    queries_df = pd.DataFrame(model_data["queries"])
    
    # Display Table with Query, Context, and Response
    st.subheader(f"Model: {selected_model}")
    st.dataframe(queries_df)
else:
    st.warning("No response data available for this model.")

# ========================== SUMMARY TABLE ==========================
st.header("Summary Table")
st.table(filtered_df)
