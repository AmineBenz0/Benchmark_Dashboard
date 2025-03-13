import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Set page config (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="LLM Metrics Dashboard", layout="wide")

# Load the aggregated data from a folder
def normalize_metric(value, min_value, max_value):
    """Normalize a metric to the range [0, 1]."""
    if max_value == min_value:
        return 0.5  # Handle edge case to avoid division by zero
    return (value - min_value) / (max_value - min_value)

@st.cache_data
def load_data_from_folder(folder_path):
    if not os.path.exists(folder_path):
        st.error(f"Folder '{folder_path}' does not exist.")
        return pd.DataFrame(), pd.DataFrame()
    
    metrics_data = []
    queries_data = []
    # Define expected ranges for normalization
    correctness_range = (0.0, 1.0)        # Correctness is already between 0 and 1
    relevance_range = (0.0, 1.0)          # Relevance is already between 0 and 1
    perplexity_range = (1.0, 10.0)        # Example range for perplexity
    resolution_time_range = (0.5, 5.0)    # Example range for resolution time

    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                model_name = filename.replace("_aggregated_results.json", "")
                try:
                    with open(os.path.join(folder_path, filename), "r") as f:
                        model_data = json.load(f)
                        # Normalize metrics
                        correctness = normalize_metric(model_data["correctness"], *correctness_range)
                        relevance = normalize_metric(model_data["relevance"], *relevance_range)
                        inverted_perplexity = 1 / model_data["perplexity"] if model_data["perplexity"] != 0 else 0
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
                except Exception as e:
                    st.error(f"Error loading {filename}: {e}")
                    continue
    except Exception as e:
        st.error(f"Error accessing folder {folder_path}: {e}")
        
    return pd.DataFrame(metrics_data), pd.DataFrame(queries_data)

@st.cache_data
def load_function_calling_data(folder_path):
    if not os.path.exists(folder_path):
        st.error(f"Function calling folder '{folder_path}' does not exist.")
        return pd.DataFrame()
    
    function_metrics = []
    
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                model_name = filename.replace(".json", "")
                try:
                    with open(os.path.join(folder_path, filename), "r") as f:
                        model_data = json.load(f)
                        metrics = {
                            "model": model_name,
                            # Key function calling metrics (non-time dependent)
                            "parsing_success_rate": model_data["aggregate_metrics"]["parsing_success_rate"],
                            "exact_function_name_rate": model_data["aggregate_metrics"]["exact_function_name_rate"],
                            "combined_function_match_rate": model_data["aggregate_metrics"]["combined_function_match_rate"],
                            "required_params_present_rate": model_data["aggregate_metrics"]["required_params_present_rate"],
                            "params_correct_type_rate": model_data["aggregate_metrics"]["params_correct_type_rate"],
                            "no_extra_params_rate": model_data["aggregate_metrics"]["no_extra_params_rate"], 
                            "exact_success_rate": model_data["aggregate_metrics"]["exact_success_rate"],
                            "combined_success_rate": model_data["aggregate_metrics"]["combined_success_rate"],
                            "correct_tool_retrieval_rate": model_data["aggregate_metrics"]["correct_tool_retrieval_rate"],
                            "correct_tool_first_rate": model_data["aggregate_metrics"]["correct_tool_first_rate"],
                            "average_functional_similarity": model_data["aggregate_metrics"]["average_functional_similarity"],
                            "average_description_similarity": model_data["aggregate_metrics"]["average_description_similarity"],
                        }
                        
                        # Calculate a combined function calling score
                        weights = {
                            "parsing_success_rate": 0.10,
                            "exact_function_name_rate": 0.15,
                            "combined_function_match_rate": 0.05,
                            "required_params_present_rate": 0.15,
                            "params_correct_type_rate": 0.10,
                            "no_extra_params_rate": 0.05, 
                            "exact_success_rate": 0.20,
                            "correct_tool_retrieval_rate": 0.10,
                            "correct_tool_first_rate": 0.10,
                        }
                        
                        function_score = sum(metrics[metric] * weight for metric, weight in weights.items())
                        metrics["function_calling_score"] = function_score
                        
                        function_metrics.append(metrics)
                except Exception as e:
                    st.error(f"Error loading function calling data {filename}: {e}")
                    continue
    except Exception as e:
        st.error(f"Error accessing function calling folder {folder_path}: {e}")
        
    return pd.DataFrame(function_metrics)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Render the dashboard for a specific dataframe
def render_dashboard(metrics_df, queries_df, function_df, title, color_scheme="Default"):
    st.title(title)
    
    # Extract unique models
    unique_models = metrics_df["model"].unique() if not metrics_df.empty else []
    unique_function_models = function_df["model"].unique() if not function_df.empty else []

    # Set color scheme based on selection
    color_map = {
        "Default": px.colors.qualitative.Plotly,
        "Blues": px.colors.sequential.Blues_r,
        "Reds": px.colors.sequential.Reds_r,
        "Greens": px.colors.sequential.Greens_r,
        "Viridis": px.colors.sequential.Viridis,
        "Plasma": px.colors.sequential.Plasma
    }
    
    colors = color_map.get(color_scheme, px.colors.qualitative.Plotly)
    
    # Add a sidebar for filters
    st.sidebar.header("Filters")
    
    # Create tabs for general metrics and function calling
    main_tab1, main_tab2 = st.tabs(["General LLM Metrics", "Function Calling Benchmark"])
    
    # == General LLM Metrics Tab ==
    with main_tab1:
        selected_models = st.sidebar.multiselect("Select models for general metrics", 
                                              unique_models, 
                                              default=unique_models)

        # Filter data based on selected models
        filtered_metrics = metrics_df[metrics_df["model"].isin(selected_models)] if not metrics_df.empty else pd.DataFrame()
        filtered_queries = queries_df[queries_df["model"].isin(selected_models)] if not queries_df.empty else pd.DataFrame()

        # Create tabs for different metric categories
        tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Efficiency Metrics", "Resource Usage"])

        # ========================== PERFORMANCE METRICS ==========================
        with tab1:
            st.header("Performance Metrics")

            col1, col2 = st.columns(2)
            
            with col1:
                # Correctness (Higher is better)
                st.subheader("Correctness (Higher is better)")
                if not filtered_metrics.empty:
                    fig1 = px.bar(filtered_metrics, x="model", y="correctness", 
                                title="Correctness Comparison", color="model", color_discrete_sequence=colors)
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("No data available for correctness")

            with col2:
                # Relevance (Higher is better)
                st.subheader("Relevance (Higher is better)")
                if not filtered_metrics.empty:
                    fig2 = px.bar(filtered_metrics, x="model", y="relevance", 
                                title="Relevance Comparison", color="model", color_discrete_sequence=colors)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No data available for relevance")

            # Combined Score (Higher is better)
            st.subheader("Combined Score (Higher is better)")
            if not filtered_metrics.empty:
                fig3 = px.bar(filtered_metrics, x="model", y="combined_score", 
                            title="Combined Score Comparison", color="model", color_discrete_sequence=colors)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No data available for combined score")

        # ========================== EFFICIENCY METRICS ==========================
        with tab2:
            st.header("Efficiency Metrics")

            col1, col2 = st.columns(2)
            
            with col1:
                # Resolution Time (Lower is better)
                st.subheader("Resolution Time (Lower is better)")
                if not filtered_metrics.empty:
                    fig4 = px.bar(filtered_metrics, x="model", y="resolution_time", 
                                title="Resolution Time Comparison", color="model", color_discrete_sequence=colors)
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("No data available for resolution time")

            with col2:
                # Perplexity (Lower is better)
                st.subheader("Perplexity (Lower is better)")
                if not filtered_metrics.empty:
                    fig5 = px.bar(filtered_metrics, x="model", y="perplexity", 
                                title="Perplexity Comparison", color="model", color_discrete_sequence=colors)
                    st.plotly_chart(fig5, use_container_width=True)
                else:
                    st.info("No data available for perplexity")

        # ========================== RESOURCE USAGE METRICS ==========================
        with tab3:
            st.header("Resource Usage Metrics")
            if not filtered_metrics.empty:
                resource_metrics = ["cpu_%", "ram_%", "gpu_mem_%"]
                resource_df = filtered_metrics.melt(id_vars=["model"], 
                                                value_vars=resource_metrics, 
                                                var_name="Metric", 
                                                value_name="Percentage")
                fig6 = px.bar(resource_df, x="model", y="Percentage", color="Metric",
                            barmode="group", title="Resource Usage Comparison",
                            color_discrete_sequence=colors[:3])
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("No resource usage data available")

        # ========================== SHOW INPUT, CONTEXT, AND OUTPUT ==========================
        st.header("📜 Model Responses Showcase")

        # Dropdown to select a single model for detailed response view
        selected_model = st.selectbox("Select a model to view responses:", 
                                    unique_models if len(unique_models) > 0 else ["No models available"])

        # Extract relevant responses for the selected model
        if not filtered_queries.empty:
            model_queries = filtered_queries[filtered_queries["model"] == selected_model][["query", "context", "final_answer"]]

            # Display Table with Query, Context, and Response
            st.subheader(f"Responses for Model: {selected_model}")
            st.dataframe(model_queries)
        else:
            st.info("No query data available")

        # ========================== SUMMARY TABLE ==========================
        st.header("Summary Table")
        if not filtered_metrics.empty:
            st.dataframe(filtered_metrics, use_container_width=True)
            
            # Add download button
            csv = convert_df_to_csv(filtered_metrics)
            st.download_button(
                "Download metrics as CSV",
                csv,
                f"llm_metrics_{title.replace(' ', '_')}.csv",
                "text/csv",
                key='download-general-csv'
            )
        else:
            st.info("No summary data available")
    
    # == Function Calling Benchmark Tab ==
    with main_tab2:
        selected_function_models = st.sidebar.multiselect("Select models for function calling", 
                                                      unique_function_models, 
                                                      default=unique_function_models)
        
        filtered_function_df = function_df[function_df["model"].isin(selected_function_models)] if not function_df.empty else pd.DataFrame()
        
        if filtered_function_df.empty:
            st.info("No function calling data available for the selected models")
        else:
            # Create tabs for different function calling metrics
            func_tab1, func_tab2, func_tab3 = st.tabs(["Function Name Metrics", "Parameter Metrics", "Overall Success"])
            
            # ========================== FUNCTION NAME METRICS ==========================
            with func_tab1:
                st.header("Function Name Recognition Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Parsing Success Rate
                    st.subheader("Parsing Success Rate")
                    fig_parse = px.bar(filtered_function_df, x="model", y="parsing_success_rate", 
                                      title="Parser Success Rate", color="model", color_discrete_sequence=colors)
                    st.plotly_chart(fig_parse, use_container_width=True)
                
                with col2:
                    # Exact Function Name Rate
                    st.subheader("Exact Function Name Match Rate")
                    fig_exact = px.bar(filtered_function_df, x="model", y="exact_function_name_rate", 
                                      title="Exact Function Name Match", color="model", color_discrete_sequence=colors)
                    st.plotly_chart(fig_exact, use_container_width=True)
                
                # Combined Function Match Rate
                st.subheader("Combined Function Match Rate")
                fig_combined = px.bar(filtered_function_df, x="model", y="combined_function_match_rate", 
                                     title="Combined Function Name Match (Exact + Fuzzy)", 
                                     color="model", color_discrete_sequence=colors)
                st.plotly_chart(fig_combined, use_container_width=True)
                
                # Function name comparison
                metrics = ["exact_function_name_rate", "combined_function_match_rate", "correct_tool_retrieval_rate", "correct_tool_first_rate"]
                labels = ["Exact Name Match", "Combined Match", "Tool Retrieval", "Correct Tool First"]
                
                compare_df = filtered_function_df.melt(id_vars=["model"], 
                                                    value_vars=metrics,
                                                    var_name="Metric", 
                                                    value_name="Rate")
                
                # Map metric names to more readable labels
                compare_df["Metric"] = compare_df["Metric"].map(dict(zip(metrics, labels)))
                
                fig_compare = px.bar(compare_df, x="model", y="Rate", color="Metric",
                                    barmode="group", title="Function Name Metrics Comparison",
                                    color_discrete_sequence=colors[:len(metrics)])
                st.plotly_chart(fig_compare, use_container_width=True)
            
            # ========================== PARAMETER METRICS ==========================
            with func_tab2:
                st.header("Parameter Handling Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Required Parameters Present
                    st.subheader("Required Parameters Present")
                    fig_req = px.bar(filtered_function_df, x="model", y="required_params_present_rate", 
                                   title="Required Parameters Presence", color="model", color_discrete_sequence=colors)
                    st.plotly_chart(fig_req, use_container_width=True)
                
                with col2:
                    # Params Correct Type
                    st.subheader("Parameters with Correct Type")
                    fig_type = px.bar(filtered_function_df, x="model", y="params_correct_type_rate", 
                                    title="Parameters with Correct Type", color="model", color_discrete_sequence=colors)
                    st.plotly_chart(fig_type, use_container_width=True)
                
                # No Extra Parameters
                st.subheader("No Extra Parameters Rate")
                fig_extra = px.bar(filtered_function_df, x="model", y="no_extra_params_rate", 
                                 title="Rate of No Extra Parameters", color="model", color_discrete_sequence=colors)
                st.plotly_chart(fig_extra, use_container_width=True)
                
                # Parameter metrics comparison
                param_metrics = ["required_params_present_rate", "params_correct_type_rate", "no_extra_params_rate"]
                param_labels = ["Required Params Present", "Correct Type", "No Extra Params"]
                
                param_df = filtered_function_df.melt(id_vars=["model"], 
                                                  value_vars=param_metrics,
                                                  var_name="Metric", 
                                                  value_name="Rate")
                
                # Map metric names to more readable labels
                param_df["Metric"] = param_df["Metric"].map(dict(zip(param_metrics, param_labels)))
                
                fig_param_compare = px.bar(param_df, x="model", y="Rate", color="Metric",
                                         barmode="group", title="Parameter Metrics Comparison",
                                         color_discrete_sequence=colors[:len(param_metrics)])
                st.plotly_chart(fig_param_compare, use_container_width=True)
            
            # ========================== OVERALL SUCCESS METRICS ==========================
            with func_tab3:
                st.header("Overall Function Calling Success")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Exact Success Rate
                    st.subheader("Exact Success Rate")
                    fig_success = px.bar(filtered_function_df, x="model", y="exact_success_rate", 
                                       title="Exact Function Call Success", color="model", color_discrete_sequence=colors)
                    st.plotly_chart(fig_success, use_container_width=True)
                
                with col2:
                    # Combined Success Rate
                    st.subheader("Combined Success Rate (Exact + Fuzzy)")
                    fig_comb_success = px.bar(filtered_function_df, x="model", y="combined_success_rate", 
                                            title="Combined Success Rate", color="model", color_discrete_sequence=colors)
                    st.plotly_chart(fig_comb_success, use_container_width=True)
                
                # Function Calling Score (custom weighted metric)
                st.subheader("Function Calling Score (Weighted Combination)")
                fig_score = px.bar(filtered_function_df, x="model", y="function_calling_score", 
                                 title="Overall Function Calling Score", color="model", color_discrete_sequence=colors)
                st.plotly_chart(fig_score, use_container_width=True)
                
                # Radar chart for overall function calling capability
                st.subheader("Function Calling Capability Overview")
                
                # Select metrics for radar chart
                radar_metrics = ["parsing_success_rate", "exact_function_name_rate", 
                               "required_params_present_rate", "params_correct_type_rate", 
                               "exact_success_rate", "combined_success_rate"]
                radar_labels = ["Parsing", "Exact Name", "Required Params", 
                              "Correct Types", "Exact Success", "Combined Success"]
                
                # Create radar chart data
                fig_radar = go.Figure()
                
                for model in filtered_function_df["model"].unique():
                    model_data = filtered_function_df[filtered_function_df["model"] == model]
                    values = [model_data[metric].values[0] for metric in radar_metrics]
                    # Add a closing value to complete the polygon
                    values.append(values[0])
                    radar_labels_with_close = radar_labels + [radar_labels[0]]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=radar_labels_with_close,
                        fill='toself',
                        name=model
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Summary Table for Function Calling
            st.header("Function Calling Summary Table")
            st.dataframe(filtered_function_df, use_container_width=True)
            
            # Add download button for function calling metrics
            func_csv = convert_df_to_csv(filtered_function_df)
            st.download_button(
                "Download function calling metrics as CSV",
                func_csv,
                f"function_calling_metrics_{title.replace(' ', '_')}.csv",
                "text/csv",
                key='download-function-csv'
            )

            # Add metric explanations
            with st.expander("Function Calling Metric Explanations"):
                st.markdown("""
                ### Function Calling Metrics Explained
                
                #### Function Name Metrics
                - **Parsing Success Rate**: How often the model's response could be parsed as a valid function call
                - **Exact Function Name Rate**: Rate of exact matches to the correct function name
                - **Combined Function Match Rate**: Rate of matches including both exact and fuzzy matches
                - **Correct Tool Retrieval Rate**: How often the model retrieved the correct tool from available options
                - **Correct Tool First Rate**: How often the correct tool was selected as the first option
                
                #### Parameter Metrics
                - **Required Parameters Present Rate**: Rate at which all required parameters are included
                - **Parameters Correct Type Rate**: Rate of parameters having the correct data type
                - **No Extra Parameters Rate**: Rate of function calls without any unnecessary extra parameters
                
                #### Success Metrics
                - **Exact Success Rate**: Rate of completely correct function calls (name + parameters)
                - **Combined Success Rate**: Success rate including both exact and fuzzy matches
                - **Function Calling Score**: Weighted combination of key metrics for overall performance
                """)

# Main function
def main():
    # Add a sidebar for selecting the LLM type and visual options
    with st.sidebar:
        st.title("LLM Benchmark Dashboard")
        st.markdown("---")
        
        st.header("Configuration")
        llm_type = st.radio("Select LLM Type", ["Small LLMs", "Big LLMs"])
        
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Default", "Blues", "Reds", "Greens", "Viridis", "Plasma"]
        )
        
        # Add explanation about the metrics
        with st.expander("About this Dashboard"):
            st.markdown("""
            ### LLM Benchmark Dashboard
            
            This dashboard provides comprehensive benchmarking for Language Models across various metrics:
            
            - **General Performance**: Correctness, relevance, perplexity, and resource usage
            - **Function Calling**: Ability to correctly identify and call functions with proper parameters
            
            The "Small LLMs" and "Big LLMs" tabs let you compare models of similar size classes.
            """)

    # Load data based on the selected LLM type
    if llm_type == "Small LLMs":
        metrics_df, queries_df = load_data_from_folder("Small_aggregated/")
        function_df = load_function_calling_data("Small_function_calling/")
        render_dashboard(metrics_df, queries_df, function_df, "Small LLM Metrics Dashboard", color_scheme)
    else:
        metrics_df, queries_df = load_data_from_folder("Big_aggregated/")
        function_df = load_function_calling_data("Big_function_calling/")
        render_dashboard(metrics_df, queries_df, function_df, "Big LLM Metrics Dashboard", color_scheme)

if __name__ == "__main__":
    main()
