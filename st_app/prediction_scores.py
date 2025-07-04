# st_app/prediction_scores.py

import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for the heatmap

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


def calculate_general_model_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the average RMSE, MAE, and MAPE for each model across all items,
    including average BMN and Min-Max Normalized scores.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing calculated metrics per model and item,
                                   including BMN and Min-Max Normalized scores.

    Returns:
        pd.DataFrame: A DataFrame with average metrics per model.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    # Define all original and normalized metric columns we expect to average
    metric_cols_to_avg = ['RMSE', 'MAE', 'MAPE', 'RMSE BMN', 'MAE BMN', 'MAPE BMN', 'RMSE MinMax Norm', 'MAE MinMax Norm', 'MAPE MinMax Norm']
    
    # Filter for columns that actually exist in the dataframe before grouping
    existing_metric_cols_to_avg = [col for col in metric_cols_to_avg if col in metrics_df.columns]

    if not existing_metric_cols_to_avg:
        return pd.DataFrame()

    # Group by model_name and calculate the mean for each metric
    general_scores_df = metrics_df.groupby('model_name')[existing_metric_cols_to_avg].mean().reset_index()
    
    # Rename columns for clarity in display
    rename_mapping = {
        'RMSE': 'RMSE AVG Model Score',
        'MAE': 'MAE AVG Model Score',
        'MAPE': 'MAPE AVG Model Score',
        'RMSE BMN': 'RMSE BMN Avg',
        'MAE BMN': 'MAE BMN Avg',
        'MAPE BMN': 'MAPE BMN Avg',
        'RMSE MinMax Norm': 'RMSE MinMax Norm Avg',
        'MAE MinMax Norm': 'MAE MinMax Norm Avg',
        'MAPE MinMax Norm': 'MAPE MinMax Norm Avg'
    }
    general_scores_df.rename(columns=rename_mapping, inplace=True)

    # Define the desired order of columns
    desired_order = [
        'model_name',
        'RMSE AVG Model Score', 'RMSE BMN Avg', 'RMSE MinMax Norm Avg',
        'MAE AVG Model Score', 'MAE BMN Avg', 'MAE MinMax Norm Avg',
        'MAPE AVG Model Score', 'MAPE BMN Avg', 'MAPE MinMax Norm Avg'
    ]
    
    # Filter and reorder columns based on what's available
    final_order = [col for col in desired_order if col in general_scores_df.columns]
    general_scores_df = general_scores_df[final_order]
    
    return general_scores_df


def heatmap_colors_style(series, cmap_name='RdYlGn_r'):
    """
    Applies a heatmap style to a pandas Series.
    Dark Green for lowest (good), Red for highest (bad). Returns a list of CSS style strings.
    """
    # Ensure series is numeric for normalization
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    if numeric_series.empty:
        return [''] * len(series)

    norm = mcolors.Normalize(vmin=numeric_series.min(), vmax=numeric_series.max())
    cmap = plt.cm.get_cmap(cmap_name)
    
    # Map original series indices to their normalized colors
    colors = [f'background-color: {mcolors.to_hex(cmap(norm(v)))}' if pd.notna(v) else '' for v in series]
    return colors

def apply_comprehensive_styling(df_to_style: pd.DataFrame, numeric_df: pd.DataFrame, metric_cols: list, odd_item_color: str = '#444444'):
    """
    Adds comprehensive styling to a DataFrame, including heatmap for metrics,
    alternating row colors for items, and item separation borders.
    This function returns a DataFrame of CSS style strings.
    
    Args:
        df_to_style (pd.DataFrame): The DataFrame to apply styles to (e.g., display_df_with_icons).
        numeric_df (pd.DataFrame): The original numeric DataFrame used for heatmap calculations.
        metric_cols (list): List of metric column names (e.g., ['RMSE', 'MAE', 'MAPE', 'RMSE BMN', ...]).
        odd_item_color (str): Hex color for alternating odd items.
                                If a dark color, text color will be set to white.
    """
    df = df_to_style

    styles_df_final = pd.DataFrame('', index=df.index, columns=df.columns)

    # Check if 'Display Name' column exists for item-specific styling (Detailed Performance table)
    if 'Display Name' in df.columns:
        unique_display_names = df['Display Name'].unique()
        for i, item_display_name in enumerate(unique_display_names):
            item_group_indices = df[df['Display Name'] == item_display_name].index
            
            # Alternating row background
            bg_color = odd_item_color if i % 2 == 0 else ''
            text_color_for_odd_rows = 'color: #FFFFFF;' if bg_color else '' 
            
            for col in df.columns:
                styles_df_final.loc[item_group_indices, col] += f"background-color: {bg_color};{text_color_for_odd_rows};"

            # Heatmap colors for metrics within this group
            for metric_col in metric_cols:
                # Ensure we only try to style columns that exist in the dataframe
                if metric_col in numeric_df.columns and not numeric_df.loc[item_group_indices, metric_col].isnull().all():
                    colors_for_metric = heatmap_colors_style(numeric_df.loc[item_group_indices, metric_col]) 
                    for j, row_idx in enumerate(item_group_indices):
                        current_style_entry = styles_df_final.loc[row_idx, metric_col]
                        current_style_entry_parts = [s for s in current_style_entry.split(';') if 'background-color' not in s and 'color' not in s]
                        final_text_color_for_heatmap = text_color_for_odd_rows 

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
    else: # For the general scores table, apply only heatmap
        for metric_col in metric_cols:
            if metric_col in numeric_df.columns and not numeric_df[metric_col].isnull().all():
                colors_for_metric = heatmap_colors_style(numeric_df[metric_col])
                for j, row_idx in enumerate(df.index):
                    styles_df_final.loc[row_idx, metric_col] = colors_for_metric[j]

    return styles_df_final

def calculate_model_ranks(metrics_df: pd.DataFrame, metrics_to_rank: list) -> pd.DataFrame:
    """
    Calculates the rank of each model for each item and specified metric.
    Lower metric score means better rank (rank 1 is best).

    Args:
        metrics_df (pd.DataFrame): DataFrame containing calculated metrics per model and item.
        metrics_to_rank (list): List of metric column names to calculate ranks for.

    Returns:
        pd.DataFrame: A DataFrame with model_name, item_id, and ranks for each metric.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    ranked_data = []
    for item_id in metrics_df['item_id'].unique():
        item_group = metrics_df[metrics_df['item_id'] == item_id].copy()
        
        for metric in metrics_to_rank:
            if metric in item_group.columns and not item_group[metric].isnull().all():
                # Rank models for the current item and metric.
                # method='min' assigns the same rank to ties.
                # ascending=True because lower scores are better.
                item_group[f'{metric}_rank'] = item_group[metric].rank(method='min', ascending=True)
            else:
                item_group[f'{metric}_rank'] = np.nan
        
        # Select relevant columns for ranks and append
        cols_to_keep = ['item_id', 'model_name'] + [f'{m}_rank' for m in metrics_to_rank]
        ranked_data.append(item_group[cols_to_keep])
    
    if ranked_data:
        return pd.concat(ranked_data, ignore_index=True)
    return pd.DataFrame()


def plot_performance_profiles(ranked_df: pd.DataFrame, metrics_to_plot: list):
    """
    Generates and displays performance profile plots for specified metrics.

    Args:
        ranked_df (pd.DataFrame): DataFrame containing model ranks per item and metric.
        metrics_to_plot (list): List of metric names for which to plot profiles.
    """
    if ranked_df.empty:
        st.info("No ranked data available to plot performance profiles.")
        return

    all_models = sorted(ranked_df['model_name'].unique().tolist())
    num_models = len(all_models)
    
    # Generate distinct colors for models
    model_colors = plt.cm.get_cmap('tab10', num_models)
    model_color_map = {model: model_colors(i) for i, model in enumerate(all_models)}

    for metric in metrics_to_plot:
        rank_col = f'{metric}_rank'
        if rank_col not in ranked_df.columns:
            st.warning(f"Rank column '{rank_col}' not found for plotting performance profile.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Determine the maximum rank to set x-axis limits
        max_rank = int(ranked_df[rank_col].max()) if not ranked_df[rank_col].isnull().all() else num_models
        if np.isnan(max_rank) or max_rank == 0: # Handle case where all ranks are NaN or only one model
            max_rank = num_models if num_models > 0 else 1

        # Create a range of ranks for the x-axis, from 1 to max_rank
        x_ranks = np.arange(1, max_rank + 1)

        for model_name in all_models:
            model_ranks = ranked_df[ranked_df['model_name'] == model_name][rank_col].dropna()
            
            if not model_ranks.empty:
                # Calculate cumulative proportion
                cumulative_proportions = [
                    (model_ranks <= r).sum() / len(model_ranks) for r in x_ranks
                ]
                ax.plot(x_ranks, cumulative_proportions, 
                        label=model_name, 
                        color=model_color_map.get(model_name, 'gray'),
                        marker='o', linestyle='-', markersize=4)

        ax.set_title(f"Performance Profile for {metric} (Lower is Better)", fontsize=16)
        ax.set_xlabel("Rank (r)", fontsize=12)
        ax.set_ylabel("Proportion of Items with Rank <= r", fontsize=12)
        ax.set_xticks(x_ranks) # Ensure integer ticks
        ax.set_ylim(0, 1.05) # Y-axis from 0 to 1 (or slightly above)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


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

    # Calculate metrics (per item and model)
    metrics_df = calculate_metrics(actuals_df, predictions_df)

    if metrics_df.empty:
        st.warning("No metrics could be calculated. This might be due to no overlapping data between predictions and actuals, or all values being NaN/zero.")
        return

    # --- Item Selection for Scores (Remains at the top for filtering) ---
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
    st.markdown("""
    **Best Model Normalized (BMN) Score Explanation:**
    The BMN score for a given metric and item indicates how a model's performance compares to the *best-performing model for that specific item*.
    
    It is calculated as:
    $$ \\text{BMN Score} = \\frac{\\text{Model's Metric Score}}{\\text{Best Model's Metric Score for this Item}} $$
    
    A BMN score of **1.0** indicates the model is the best performer for that item and metric. A score greater than 1.0 means the model is worse than the best, with higher values indicating poorer relative performance. This metric helps to understand a model's performance on a relative scale for each individual time series.

    **Min-Max Normalized Score Explanation:**
    The Min-Max Normalized score scales a model's performance for a given metric and item into a range between 0 and 1.
    
    It is calculated as:
    $$ \\text{Min-Max Normalized Score} = \\frac{\\text{Model's Metric Score} - \\text{Minimum Metric Score for this Item}}{\\text{Maximum Metric Score for this Item} - \\text{Minimum Metric Score for this Item}} $$
    
    A score of **0.0** represents the best performance for that item and metric (the minimum score), while **1.0** represents the worst performance (the maximum score). This normalization is useful for understanding a model's performance relative to the full range of observed performances for a specific item.
    """)

    # Add display names to the filtered_metrics_df for better readability
    if 'Display Name' not in filtered_metrics_df.columns:
        filtered_metrics_df['Display Name'] = filtered_metrics_df['item_id'].apply(lambda x: get_display_name_func(x, item_descriptions_df))
    
    # --- Calculate BMN and Min-Max Norm scores ---
    extended_metrics_df = filtered_metrics_df.copy()
    metrics_for_normalization = ['RMSE', 'MAE', 'MAPE']

    for item_id in extended_metrics_df['item_id'].unique():
        item_slice_indices = extended_metrics_df[extended_metrics_df['item_id'] == item_id].index
        
        for metric in metrics_for_normalization:
            item_metric_series = extended_metrics_df.loc[item_slice_indices, metric].dropna()

            if not item_metric_series.empty:
                # Calculate BMN
                best_score = item_metric_series.min()
                if best_score == 0: # Avoid division by zero if best score is 0
                    extended_metrics_df.loc[item_slice_indices, f'{metric} BMN'] = np.nan
                else:
                    extended_metrics_df.loc[item_slice_indices, f'{metric} BMN'] = extended_metrics_df.loc[item_slice_indices, metric] / best_score
                
                # Calculate Min-Max Norm
                min_score = item_metric_series.min()
                max_score = item_metric_series.max()

                if max_score == min_score: # Avoid division by zero if all scores are identical
                    extended_metrics_df.loc[item_slice_indices, f'{metric} MinMax Norm'] = 0.0 # All models are equally "best"
                else:
                    extended_metrics_df.loc[item_slice_indices, f'{metric} MinMax Norm'] = (extended_metrics_df.loc[item_slice_indices, metric] - min_score) / (max_score - min_score)
            else: # No valid metrics for this item
                extended_metrics_df.loc[item_slice_indices, f'{metric} BMN'] = np.nan
                extended_metrics_df.loc[item_slice_indices, f'{metric} MinMax Norm'] = np.nan
    
    # Reorder columns for display and drop original index
    # Removed 'item_id' here
    display_cols = ['Display Name', 'model_name']
    for metric in metrics_for_normalization:
        display_cols.append(metric)
        display_cols.append(f'{metric} BMN')
        display_cols.append(f'{metric} MinMax Norm')

    metrics_df_display = extended_metrics_df[display_cols].copy()
    metrics_df_display = metrics_df_display.reset_index(drop=True) # Remove the default index

    # Sort for better presentation - crucial for border logic
    metrics_df_display.sort_values(by=['Display Name', 'model_name'], inplace=True)
    
    # Define all metric columns that should get heatmap coloring (original, BMN, and MinMax Norm)
    metric_columns_to_heatmap = metrics_for_normalization + \
                                [f'{m} BMN' for m in metrics_for_normalization] + \
                                [f'{m} MinMax Norm' for m in metrics_for_normalization]

    # --- Icon and Formatting Logic ---
    # Create a DataFrame for display that includes icons
    display_df_with_icons = metrics_df_display.copy()

    unique_display_names_for_icons = display_df_with_icons['Display Name'].unique()
    for item_display_name in unique_display_names_for_icons:
        item_group_numeric_for_min = metrics_df_display[metrics_df_display['Display Name'] == item_display_name] # Use numeric for min find

        for metric_col in metrics_for_normalization: # Iterate original metrics for their stars
            if metric_col in item_group_numeric_for_min.columns and not item_group_numeric_for_min[metric_col].isnull().all():
                min_val = item_group_numeric_for_min[metric_col].min()
                min_val_rows_indices = item_group_numeric_for_min[item_group_numeric_for_min[metric_col] == min_val].index
                
                for idx in min_val_rows_indices:
                    current_value = metrics_df_display.loc[idx, metric_col] # Get original numeric value
                    formatted_value = f"{current_value:.2f}" if metric_col != 'MAPE' else f"{current_value:.2f}%"
                    display_df_with_icons.loc[idx, metric_col] = f"⭐ {formatted_value}" # Add star icon

        # Add star icons for BMN columns (where value is ~1.0)
        for bmn_metric_col in [f'{m} BMN' for m in metrics_for_normalization]:
            if bmn_metric_col in item_group_numeric_for_min.columns and not item_group_numeric_for_min[bmn_metric_col].isnull().all():
                min_bmn_val = item_group_numeric_for_min[bmn_metric_col].min()
                # Use np.isclose for float comparisons, best BMN is 1.0 (or very close)
                best_bmn_rows_indices = item_group_numeric_for_min[np.isclose(item_group_numeric_for_min[bmn_metric_col], min_bmn_val)].index
                
                for idx in best_bmn_rows_indices:
                    current_value = metrics_df_display.loc[idx, bmn_metric_col]
                    formatted_value = f"{current_value:.2f}"
                    display_df_with_icons.loc[idx, bmn_metric_col] = f"⭐ {formatted_value}"

        # Add star icons for MinMax Norm columns (where value is ~0.0)
        for minmax_metric_col in [f'{m} MinMax Norm' for m in metrics_for_normalization]:
            if minmax_metric_col in item_group_numeric_for_min.columns and not item_group_numeric_for_min[minmax_metric_col].isnull().all():
                min_minmax_val = item_group_numeric_for_min[minmax_metric_col].min()
                # Use np.isclose for float comparisons, best MinMax Norm is 0.0 (or very close)
                best_minmax_rows_indices = item_group_numeric_for_min[np.isclose(item_group_numeric_for_min[minmax_metric_col], min_minmax_val)].index
                
                for idx in best_minmax_rows_indices:
                    current_value = metrics_df_display.loc[idx, minmax_metric_col]
                    formatted_value = f"{current_value:.2f}"
                    display_df_with_icons.loc[idx, minmax_metric_col] = f"⭐ {formatted_value}"
    
    # Apply comprehensive styling
    styled_df = display_df_with_icons.style.apply(
        apply_comprehensive_styling,
        numeric_df=metrics_df_display,
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

    st.write("---")

    # --- Models General Score - Metric Avg ---
    st.subheader("Models General Score - Metric Avg")

    # Pass the extended_metrics_df (which now includes BMN and MinMax Norm)
    general_scores_df = calculate_general_model_scores(extended_metrics_df)

    if not general_scores_df.empty:
        # Define all metric columns for heatmap coloring for general scores
        # This list should match the `final_order` from calculate_general_model_scores
        general_metric_columns_to_heatmap = [
            'RMSE AVG Model Score', 'RMSE BMN Avg', 'RMSE MinMax Norm Avg',
            'MAE AVG Model Score', 'MAE BMN Avg', 'MAE MinMax Norm Avg',
            'MAPE AVG Model Score', 'MAPE BMN Avg', 'MAPE MinMax Norm Avg'
        ]
        
        # Create a display DataFrame for general scores to add icons and formatting
        general_scores_display = general_scores_df.copy()

        # Add star icons to the best (lowest) average score for each metric
        for col in general_metric_columns_to_heatmap:
            if not general_scores_df[col].isnull().all():
                min_val = general_scores_df[col].min()
                # Find rows with the minimum value and add star icon
                # Use np.isclose for float comparisons, especially for normalized scores where 0.0 or 1.0 is exact
                best_min_rows_indices = general_scores_df[np.isclose(general_scores_df[col], min_val)].index
                
                for idx in best_min_rows_indices:
                    current_value = general_scores_df.loc[idx, col]
                    formatted_value = f"{current_value:.2f}"
                    if 'MAPE' in col and 'Avg' not in col: # Only apply % to raw MAPE, not its normalized versions
                        formatted_value += "%"
                    general_scores_display.loc[idx, col] = f"⭐ {formatted_value}"
            
            # Ensure other values are formatted without icon
            for idx in general_scores_df.index:
                # Only format if it's still a number (not already formatted with a star)
                if isinstance(general_scores_display.loc[idx, col], (int, float)): 
                    current_value = general_scores_display.loc[idx, col]
                    formatted_value = f"{current_value:.2f}"
                    if 'MAPE' in col and 'Avg' not in col:
                        formatted_value += "%"
                    general_scores_display.loc[idx, col] = formatted_value
                elif pd.isna(general_scores_display.loc[idx, col]): # Handle NaN
                    general_scores_display.loc[idx, col] = "N/A"


        styled_general_scores_df = general_scores_display.style.apply(
            apply_comprehensive_styling,
            numeric_df=general_scores_df, # Use the original numeric DataFrame for heatmap calculation
            metric_cols=general_metric_columns_to_heatmap,
            axis=None
        )
        st.dataframe(styled_general_scores_df, use_container_width=True)
    else:
        st.info("No general model scores could be calculated.")

    st.write("---")

    # --- Heatmap Section ---
    st.subheader("General Model Performance overview - Min Max Normalization Heatmap")

    # Filter data for selected models and items for the heatmap
    # We need the MinMax Norm score for each metric (RMSE, MAE, MAPE)
    # Let's use the average of these MinMax Norm scores for the heatmap value
    heatmap_data_raw = extended_metrics_df[
        extended_metrics_df['item_id'].isin(filtered_metrics_df['item_id'].unique())
    ].copy()

    if heatmap_data_raw.empty:
        st.info("No data available for the heatmap based on selected items.")
        return

    # Calculate the average MinMax Norm score across RMSE, MAE, MAPE for each model-item pair
    heatmap_data_raw['Avg MinMax Norm Score'] = heatmap_data_raw[[
        f'{m} MinMax Norm' for m in metrics_for_normalization
    ]].mean(axis=1)

    # Pivot the data for the heatmap: models as index, items as columns, Avg MinMax Norm Score as values
    heatmap_pivot = heatmap_data_raw.pivot_table(
        index='model_name',
        columns='item_id',
        values='Avg MinMax Norm Score'
    )

    # Map item_id columns to their display names for better readability on the heatmap
    # Ensure columns are sorted for consistent display
    heatmap_pivot = heatmap_pivot[sorted(heatmap_pivot.columns.tolist())]

    display_column_names = {
        item_id: get_display_name_func(item_id, item_descriptions_df)
        for item_id in heatmap_pivot.columns
    }
    heatmap_pivot = heatmap_pivot.rename(columns=display_column_names)

    # Sort models (index) alphabetically for consistent display
    heatmap_pivot = heatmap_pivot.sort_index()

    if not heatmap_pivot.empty:
        # Adjust figsize dynamically based on the number of items and models for better visualization
        num_items = heatmap_pivot.shape[1]
        num_models = heatmap_pivot.shape[0]
        
        # Base figure size, adjusted factors for width and height
        fig_width = max(8, num_items * 0.8) 
        fig_height = max(6, num_models * 0.8) 

        fig_heatmap, ax_heatmap = plt.subplots(figsize=(fig_width, fig_height))
        
        sns.heatmap(
            heatmap_pivot,
            annot=False, # Values are not displayed on the heatmap itself, as requested
            cmap='RdYlGn_r', # Changed color map to RdYlGn_r (Green is good/low value)
            fmt=".2f", # Format for annotations if they were enabled (not used here)
            linewidths=.5, # Lines between cells
            ax=ax_heatmap,
            cbar_kws={'label': 'Average Min-Max Normalization Score'} # Label for the color bar
        )
        ax_heatmap.set_title("General Model Performance overview - Min Max Normalization Heatmap", fontsize=14)
        ax_heatmap.set_xlabel("Item (Series Display Name)", fontsize=12)
        ax_heatmap.set_ylabel("Model Name", fontsize=12)

        # Make x-axis labels (item display names) smaller and rotate for better fit
        plt.xticks(rotation=90, ha='center', fontsize=8) 
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        st.pyplot(fig_heatmap)
        plt.close(fig_heatmap)
    else:
        st.info("No aggregated data to display in the heatmap.")

    st.write("---")

    # --- Performance Profiles (Dominance Plots) Section ---
    st.subheader("Model Performance Profiles (Dominance Plots)")
    st.markdown("""
    Performance profiles visualize how consistently each model performs across different items.
    For each metric, the plot shows the proportion of items for which a model achieved a rank
    less than or equal to a given rank (on the x-axis).
    
    A curve that rises quickly and stays high indicates a model that consistently achieves
    good ranks across many items.
    """)

    metrics_to_plot_profiles = ['RMSE', 'MAE', 'MAPE']
    
    # Calculate ranks for all relevant metrics
    ranked_metrics_df = calculate_model_ranks(extended_metrics_df, metrics_to_plot_profiles)

    if not ranked_metrics_df.empty:
        plot_performance_profiles(ranked_metrics_df, metrics_to_plot_profiles)
    else:
        st.info("Cannot generate performance profiles as no valid ranks could be calculated.")
