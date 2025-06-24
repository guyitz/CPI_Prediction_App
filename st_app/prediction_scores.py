# st_app/prediction_scores.py

import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt # Added: Import for color mapping functions

def calculate_metrics(actuals_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various prediction performance metrics (RMSE, MAE, MAPE)
    for each item_id and model_name.

    Args:
        actuals_df (pd.DataFrame): DataFrame containing actual time series data.
                                   Expected to have 'timestamp', 'item_id', 'target'.
        predictions_df (pd.DataFrame): DataFrame containing prediction data.
                                    Expected to have 'timestamp', 'item_id', 'model_name', 'mean'.

    Returns:
        pd.DataFrame: A DataFrame with calculated metrics per model and item.
    """
    if actuals_df.empty or predictions_df.empty:
        return pd.DataFrame()

    # Merge actuals and predictions on timestamp and item_id
    # Ensure 'timestamp' is a column in actuals_df for merging
    actuals_reset = actuals_df.reset_index()
    merged_df = pd.merge(
        predictions_df,
        actuals_reset[['timestamp', 'item_id', 'target']],
        on=['timestamp', 'item_id'],
        how='left'
    )

    # Drop rows where actuals are missing (i.e., predictions beyond available actuals)
    merged_df.dropna(subset=['target', 'mean'], inplace=True)

    if merged_df.empty:
        return pd.DataFrame()

    results = []
    # Group by item_id and model_name to calculate metrics for each series-model combination
    for (item_id, model_name), group in merged_df.groupby(['item_id', 'model_name']):
        y_true = group['target']
        y_pred = group['mean']

        if len(y_true) > 0 and len(y_pred) > 0:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate MAPE, handling division by zero
            # Only consider non_zero actuals for MAPE calculation to avoid inf/NaN
            non_zero_actuals_mask = y_true != 0
            if non_zero_actuals_mask.any():
                mape_values = np.abs((y_true[non_zero_actuals_mask] - y_pred[non_zero_actuals_mask]) / y_true[non_zero_actuals_mask])
                mape = np.mean(mape_values) * 100 if len(mape_values) > 0 else np.nan
            else:
                mape = np.nan # Cannot calculate MAPE if all actuals are zero

            results.append({
                'item_id': item_id,
                'model_name': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            })

    metrics_df = pd.DataFrame(results)
    return metrics_df


def heatmap_colors_style(series, cmap_name='coolwarm_r'): # Blue (low/best) to Red (high/worst)
    """
    Applies a heatmap style to a pandas Series.
    Blue for lowest, Red for highest. Returns a list of CSS style strings.
    """
    # Ensure series is numeric for normalization
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    if numeric_series.empty:
        return [''] * len(series) # Return empty styles if no numeric data

    norm = mcolors.Normalize(vmin=numeric_series.min(), vmax=numeric_series.max())
    cmap = plt.cm.get_cmap(cmap_name)
    
    # Map original series indices to their normalized colors
    colors = [f'background-color: {mcolors.to_hex(cmap(norm(v)))}' if pd.notna(v) else '' for v in series]
    return colors

def apply_comprehensive_styling(df_to_style: pd.DataFrame, numeric_df: pd.DataFrame, metric_cols: list, odd_item_color: str = '#444444'):
    """
    Applies heatmap coloring for metrics, alternating row colors for items, and item separation borders.
    This function returns a DataFrame of CSS style strings.
    
    Args:
        df_to_style (pd.DataFrame): The DataFrame to apply styles to (e.g., display_df_with_icons).
        numeric_df (pd.DataFrame): The original numeric DataFrame used for heatmap calculations.
        metric_cols (list): List of metric column names (e.g., ['RMSE', 'MAE', 'MAPE']).
        odd_item_color (str): Hex color for alternating odd items.
                                If a dark color, text color will be set to white.
    """
    df = df_to_style

    styles_df_final = pd.DataFrame('', index=df.index, columns=df.columns)

    unique_display_names = df['Display Name'].unique()
    for i, item_display_name in enumerate(unique_display_names):
        item_group_indices = df[df['Display Name'] == item_display_name].index
        
        # Alternating row background
        # Even index (0, 2, ...) gets color, odd doesn't (appears as transparent/browser default)
        bg_color = odd_item_color if i % 2 == 0 else ''
        
        # If a background color is applied (i.e., for odd items), set text color to white for better contrast
        # This helps readability regardless of Streamlit's light/dark mode.
        text_color_for_odd_rows = 'color: #FFFFFF;' if bg_color else '' 
        
        for col in df.columns:
            # Append background and text color styles
            styles_df_final.loc[item_group_indices, col] += f"background-color: {bg_color};{text_color_for_odd_rows};"

        # Heatmap colors for metrics within this group
        for metric_col in metric_cols:
            if metric_col in numeric_df.columns and not numeric_df.loc[item_group_indices, metric_col].isnull().all():
                colors_for_metric = heatmap_colors_style(numeric_df.loc[item_group_indices, metric_col], cmap_name='coolwarm_r')
                for j, row_idx in enumerate(item_group_indices):
                    current_style_entry = styles_df_final.loc[row_idx, metric_col]
                    # Remove any previous background-color before appending new one from heatmap
                    # Ensure original text color from alternating row is preserved if needed,
                    # or explicitly override with black if heatmap is applied to light background
                    current_style_entry_parts = [s for s in current_style_entry.split(';') if 'background-color' not in s and 'color' not in s]
                    
                    # Decide text color for heatmap cells:
                    # If the heatmap color is dark, text should be light. If light, text should be dark.
                    # This is complex without knowing the specific heatmap color.
                    # For simplicity, we assume Streamlit's default text color is usually good on heatmap,
                    # unless the odd/even row background explicitly set it to white.
                    # We will ensure the white text set for odd rows persists here.
                    final_text_color_for_heatmap = text_color_for_odd_rows # Keep the white text if it's an odd row

                    styles_df_final.loc[row_idx, metric_col] = f"{';'.join(current_style_entry_parts)}{colors_for_metric[j]};{final_text_color_for_heatmap}"

    # Apply item separation borders (top border for new items)
    new_item_mask = (df['Display Name'] != df['Display Name'].shift(1)).fillna(False)
    if not df.empty:
        if df.index[0] in new_item_mask.index and new_item_mask.iloc[0]:
            new_item_mask.loc[df.index[0]] = False # Do not apply border to the very first row
    
    for idx in df.index:
        if new_item_mask.loc[idx]:
            for col in df.columns:
                styles_df_final.loc[idx, col] += 'border-top: 2px solid #555555;'
    
    return styles_df_final


def render_prediction_score_section(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame, item_descriptions_df: pd.DataFrame, get_display_name_func):
    """
    Renders the Prediction Result Score section of the Streamlit app.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing prediction data.
        actuals_df (pd.DataFrame): DataFrame containing actual time series data.
        item_descriptions_df (pd.DataFrame): DataFrame containing item descriptions for mapping.
        get_display_name_func (function): A function to get the display name for an item_id.
    """
    st.header("📊 Prediction Result Score")

    if predictions_df.empty or actuals_df.empty:
        st.info("No prediction data or actuals data available. Please ensure a results folder is selected in the sidebar and data is loaded.")
        return

    st.write("This section displays the performance metrics for your forecast models.")

    # Calculate metrics
    metrics_df = calculate_metrics(actuals_df, predictions_df)

    if metrics_df.empty:
        st.warning("No metrics could be calculated. This might be due to no overlapping data between predictions and actuals, or all values being NaN/zero.")
        return

    # --- Item Selection for Scores ---
    st.subheader("Select Item(s) for Score Analysis")
    all_items_original_scores = metrics_df['item_id'].unique().tolist()
    
    # Create display names for multiselect in the format "New Name (Original ID)"
    all_items_display_scores_formatted = []
    for item_id_original in all_items_original_scores:
        display_name = get_display_name_func(item_id_original, item_descriptions_df)
        if display_name != item_id_original: # If a mapping exists
            all_items_display_scores_formatted.append(f"{display_name} ({item_id_original})")
        else: # If no mapping exists, use original ID directly
            all_items_display_scores_formatted.append(item_id_original)

    # Add 'All Items' option
    all_items_options_with_all = ['All Items'] + sorted(all_items_display_scores_formatted)

    selected_items_display_scores = st.multiselect(
        "Choose items to display scores for:",
        options=all_items_options_with_all,
        default=['All Items'] # Default to 'All Items'
    )

    # Filter metrics_df based on selection
    filtered_metrics_df = metrics_df.copy()
    if 'All Items' not in selected_items_display_scores:
        selected_items_original_scores = []
        for display_name_selected in selected_items_display_scores:
            if "(" in display_name_selected and ")" in display_name_selected:
                original_id = display_name_selected.split('(')[-1].strip(')')
            else:
                original_id = display_name_selected
            selected_items_original_scores.append(original_id)
        filtered_metrics_df = metrics_df[filtered_metrics_df['item_id'].isin(selected_items_original_scores)].copy()
    
    if filtered_metrics_df.empty:
        st.info("No scores to display for the selected item(s).")
        return

    st.write("---")

    # Detailed Performance Per Item and Model
    st.subheader("Detailed Performance Per Item and Model")

    # Add display names to the filtered_metrics_df for better readability
    if 'Display Name' not in filtered_metrics_df.columns:
        filtered_metrics_df['Display Name'] = filtered_metrics_df['item_id'].apply(lambda x: get_display_name_func(x, item_descriptions_df))
    
    # Reorder columns for display and drop original index
    display_cols = ['Display Name', 'item_id', 'model_name', 'RMSE', 'MAE', 'MAPE']
    metrics_df_display = filtered_metrics_df[display_cols].copy()
    metrics_df_display = metrics_df_display.reset_index(drop=True) # Remove the default index

    # Sort for better presentation - crucial for border logic
    metrics_df_display.sort_values(by=['Display Name', 'model_name'], inplace=True)
    
    # Define metric columns for heatmap coloring
    metric_columns_to_heatmap = ['RMSE', 'MAE', 'MAPE']

    # --- Icon and Formatting Logic ---
    # Create a DataFrame for display that includes icons
    # This DataFrame will hold string representations of values, with icons prepended
    display_df_with_icons = metrics_df_display.copy()

    unique_display_names_for_icons = display_df_with_icons['Display Name'].unique()
    for item_display_name in unique_display_names_for_icons:
        item_group_numeric_for_min = metrics_df_display[metrics_df_display['Display Name'] == item_display_name] # Use numeric for min find

        for metric_col in metric_columns_to_heatmap:
            if metric_col in item_group_numeric_for_min.columns and not item_group_numeric_for_min[metric_col].isnull().all():
                min_val = item_group_numeric_for_min[metric_col].min()
                # Find the index in the original (or reset_index) df where this min value occurs
                min_val_rows_indices = item_group_numeric_for_min[item_group_numeric_for_min[metric_col] == min_val].index
                
                for idx in min_val_rows_indices:
                    current_value = metrics_df_display.loc[idx, metric_col] # Get original numeric value
                    # Format value before adding icon, so icon is clearly separate
                    formatted_value = f"{current_value:.2f}" if metric_col != 'MAPE' else f"{current_value:.2f}%"
                    display_df_with_icons.loc[idx, metric_col] = f"⭐ {formatted_value}" # Add star icon

    # Apply comprehensive styling
    # The `apply_comprehensive_styling` function will now work with `display_df_with_icons`
    # for row backgrounds/borders, but it needs `metrics_df_display` (numeric)
    # for correct heatmap gradient calculation.
    styled_df = display_df_with_icons.style.apply(
        apply_comprehensive_styling, # Directly pass the function
        numeric_df=metrics_df_display, # Pass numeric_df as a keyword argument
        metric_cols=metric_columns_to_heatmap,
        axis=None
    )
    
    st.dataframe(styled_df, use_container_width=True)

    st.write("---")

    st.subheader("Best and Average Performing Models by Metric Per Item")

    per_item_summary_data = []
    # Iterate through each unique item in the filtered data and build its summary row
    unique_items_filtered = filtered_metrics_df['item_id'].unique()
    for item_id in sorted(unique_items_filtered):
        item_display_name = get_display_name_func(item_id, item_descriptions_df)
        item_group_df = filtered_metrics_df[filtered_metrics_df['item_id'] == item_id]

        row_data = {
            'Item Display Name': item_display_name
        }

        # RMSE
        if not item_group_df['RMSE'].isnull().all():
            best_rmse_val = item_group_df['RMSE'].min()
            best_rmse_model_name = item_group_df.loc[item_group_df['RMSE'].idxmin()]['model_name']
            avg_rmse_val = item_group_df['RMSE'].mean()
            row_data['RMSE Best Model'] = best_rmse_model_name
            row_data['RMSE Best Value'] = f"{best_rmse_val:.2f}"
            row_data['RMSE Avg'] = f"{avg_rmse_val:.2f}"
        else:
            row_data['RMSE Best Model'] = "N/A"
            row_data['RMSE Best Value'] = "N/A"
            row_data['RMSE Avg'] = "N/A"

        # MAE
        if not item_group_df['MAE'].isnull().all():
            best_mae_val = item_group_df['MAE'].min()
            best_mae_model_name = item_group_df.loc[item_group_df['MAE'].idxmin()]['model_name']
            avg_mae_val = item_group_df['MAE'].mean()
            row_data['MAE Best Model'] = best_mae_model_name
            row_data['MAE Best Value'] = f"{best_mae_val:.2f}"
            row_data['MAE Avg'] = f"{avg_mae_val:.2f}"
        else:
            row_data['MAE Best Model'] = "N/A"
            row_data['MAE Best Value'] = "N/A"
            row_data['MAE Avg'] = "N/A"

        # MAPE
        mape_filtered_item_df = item_group_df.dropna(subset=['MAPE'])
        if not mape_filtered_item_df.empty and not mape_filtered_item_df['MAPE'].isnull().all():
            best_mape_val = mape_filtered_item_df['MAPE'].min()
            best_mape_model_name = mape_filtered_item_df.loc[mape_filtered_item_df['MAPE'].idxmin()]['model_name']
            avg_mape_val = mape_filtered_item_df['MAPE'].mean()
            row_data['MAPE Best Model'] = best_mape_model_name
            row_data['MAPE Best Value'] = f"{best_mape_val:.2f}%"
            row_data['MAPE Avg'] = f"{avg_mape_val:.2f}%"
        else:
            row_data['MAPE Best Model'] = "N/A"
            row_data['MAPE Best Value'] = "N/A"
            row_data['MAPE Avg'] = "N/A"
        
        per_item_summary_data.append(row_data)
    
    if per_item_summary_data:
        per_item_summary_df = pd.DataFrame(per_item_summary_data)
        st.dataframe(per_item_summary_df.set_index('Item Display Name'), use_container_width=True)
    else:
        st.info("No per-item summary data could be generated for the selected items.")
