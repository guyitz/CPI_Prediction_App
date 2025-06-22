# st_app/prediction_plots.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# This module will need access to data loading functions and item display name function
# We will pass these as arguments from streamlit_app.py
# No direct imports from data_loader here, as they are passed in for better modularity.

def render_prediction_plots_section(actuals_df: pd.DataFrame, predictions_df: pd.DataFrame, item_descriptions_df: pd.DataFrame, get_display_name_func):
    """
    Renders the Prediction Result Plots section of the Streamlit app.

    Args:
        actuals_df (pd.DataFrame): DataFrame containing actual time series data.
        predictions_df (pd.DataFrame): DataFrame containing prediction data.
        item_descriptions_df (pd.DataFrame): DataFrame containing item descriptions for mapping.
        get_display_name_func (function): A function to get the display name for an item_id.
    """
    st.header("📈 Prediction Result Plots")
    
    if predictions_df.empty or actuals_df.empty:
        st.info("Please select a valid results folder in the sidebar and ensure data files are present.")
        return

    # 1. Model Selection (Checkboxes)
    st.sidebar.subheader("2. Select Models")
    all_models = predictions_df['model_name'].unique().tolist()
    selected_models = st.sidebar.multiselect(
        "Choose models to display:",
        options=sorted(all_models),
        default=sorted(all_models) # Select all by default
    )

    # 2. Item Selection (using the same item selection logic as EDA for consistency)
    st.sidebar.subheader("3. Select Items")
    all_items_original = predictions_df['item_id'].unique().tolist()
    
    all_items_display_formatted_plots = []
    for item_id_original in all_items_original:
        display_name = get_display_name_func(item_id_original, item_descriptions_df)
        if display_name != item_id_original:
            all_items_display_formatted_plots.append(f"{display_name} ({item_id_original})")
        else:
            all_items_display_formatted_plots.append(item_id_original)

    selected_items_display = st.sidebar.multiselect(
        "Choose items to display plots for:",
        options=sorted(all_items_display_formatted_plots),
        default=sorted(all_items_display_formatted_plots) # Select all by default
    )
    
    selected_items_original = []
    for display_name_selected in selected_items_display:
        if "(" in display_name_selected and ")" in display_name_selected:
            original_id = display_name_selected.split('(')[-1].strip(')')
        else:
            original_id = display_name_selected
        selected_items_original.append(original_id)


    if not selected_models:
        st.warning("Please select at least one model to display predictions.")
        return
    if not selected_items_original:
        st.warning("Please select at least one item to display plots.")
        return

    # If models and items are selected, proceed to plot
    # Determine the start date for actuals (last 10 years relative to the latest actual timestamp)
    # Find the max timestamp in actuals_df for the selected items
    max_actual_timestamp = actuals_df[actuals_df['item_id'].isin(selected_items_original)].index.max()
    if pd.isna(max_actual_timestamp):
        st.warning("No actual data found for the selected items to determine the 'last 10 years'. Displaying all available actuals.")
        start_date_actuals = actuals_df.index.min()
    else:
        start_date_actuals = max_actual_timestamp - pd.DateOffset(years=10)

    # Generate distinct colors for models
    model_colors = plt.cm.get_cmap('tab10', len(all_models))
    model_color_map = {model: model_colors(i) for i, model in enumerate(all_models)}

    for item_id_original in selected_items_original:
        display_item_name = get_display_name_func(item_id_original, item_descriptions_df)
        fig, ax = plt.subplots(figsize=(12, 6))

        item_actuals = actuals_df.loc[
            (actuals_df['item_id'] == item_id_original) &
            (actuals_df.index >= start_date_actuals)
        ].reset_index()

        if not item_actuals.empty:
            ax.plot(item_actuals['timestamp'], item_actuals['target'],
                    label='Actual', color='black', linewidth=2)
        else:
            st.warning(f"No actual data available for '{display_item_name}' in the last 10 years. Showing only predictions if available.")

        for model_name in sorted(selected_models):
            model_preds = predictions_df[
                (predictions_df['item_id'] == item_id_original) &
                (predictions_df['model_name'] == model_name)
            ]

            if not model_preds.empty:
                ax.plot(model_preds['timestamp'], model_preds['mean'],
                        label=f'Predicted ({model_name})',
                        color=model_color_map.get(model_name, 'gray'),
                        linestyle='--', alpha=0.8)

            else:
                st.info(f"No prediction data for '{model_name}' for item '{display_item_name}'.")

        ax.set_title(f"Forecast vs Actual for {display_item_name}", fontsize=16)
        ax.set_xlabel("Timestamp", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        
        ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

