# st_app/comparison_section.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pathlib import Path

# Assuming calculate_metrics is available from prediction_scores.py
# This requires prediction_scores.py to be in the same st_app directory
from prediction_scores import calculate_metrics

def heatmap_colors_diff(series, cmap_name='seismic'): # Blue (positive, B better) to Red (negative, A better)
    """
    Applies a heatmap style to a pandas Series, specifically for percentage differences.
    Uses 'seismic' colormap where blue is good (B better) and red is bad (A better on errors).
    """
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    if numeric_series.empty:
        return [''] * len(series)

    # Ensure min/max for normalization covers negative and positive ranges symmetrically
    # For 'seismic', 0 is white, positive values are blue, negative are red.
    # We want positive (B better) to be blue and negative (A better) to be red.
    # The default 'seismic' already aligns with this: positive (good diff) is blue, negative (bad diff) is red.
    abs_max = max(abs(numeric_series.min()), abs(numeric_series.max()))
    if abs_max == 0: # Handle case where all diffs are zero
        return [f'background-color: {mcolors.to_hex(plt.cm.get_cmap(cmap_name)(0.5))}' for _ in series] # Midpoint color

    norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max) # Symmetrical normalization
    
    cmap = plt.cm.get_cmap(cmap_name)
    colors = [f'background-color: {mcolors.to_hex(cmap(norm(v)))}' if pd.notna(v) else '' for v in series]
    return colors

def apply_comparison_styling(styled_df_data: pd.DataFrame, numeric_df_for_heatmap: pd.DataFrame, odd_item_color: str = '#2C3E50') -> pd.DataFrame: # Darker Gray/Blue
    """
    Applies alternating row colors for items and heatmap for percentage differences.
    If a dark `odd_item_color` is used, the text color for those rows will be set to white
    to ensure readability in both light and dark Streamlit themes.
    """
    df = styled_df_data # The DataFrame to style (which includes string formatting and icons)
    styles_df_final = pd.DataFrame('', index=df.index, columns=df.columns)
    
    unique_display_names = df['Item Display Name'].unique() # Assuming this column exists for grouping
    
    # Apply alternating item background colors
    for i, item_display_name in enumerate(unique_display_names):
        item_group_indices = df[df['Item Display Name'] == item_display_name].index
        
        # Even index (0, 2, ...) gets color, odd doesn't (appears as transparent/browser default)
        bg_color = odd_item_color if i % 2 == 0 else '' 

        # If a dark background color is applied (for odd items), set text color to white for better contrast.
        # This ensures readability regardless of Streamlit's light/dark mode.
        # Check if the background color is explicitly set and is dark.
        # A simple heuristic: if the first char is '#' and it's not empty, assume it's a dark color.
        is_dark_bg = bool(bg_color)
        text_color_for_odd_rows = 'color: #FFFFFF;' if is_dark_bg else ''
        
        for col in df.columns:
            # Append background and text color styles
            styles_df_final.loc[item_group_indices, col] += f"background-color: {bg_color};{text_color_for_odd_rows};"

    # Apply heatmap for percentage difference columns
    diff_cols = [col for col in df.columns if 'Diff %' in col]
    for diff_col in diff_cols:
        # Get the numeric data for the heatmap calculation
        numeric_diff_series = numeric_df_for_heatmap[diff_col]
        if not numeric_diff_series.empty and not numeric_diff_series.isnull().all():
            colors_for_diff = heatmap_colors_diff(numeric_diff_series, cmap_name='seismic') # Blue (positive) to Red (negative)
            for row_idx, style in zip(numeric_diff_series.index, colors_for_diff):
                current_style_entry = styles_df_final.loc[row_idx, diff_col]
                # Remove previous background-color and color if any, then append heatmap color
                current_style_entry_parts = [s for s in current_style_entry.split(';') if 'background-color' not in s and 'color' not in s]
                
                # Determine final text color for heatmap cells.
                # If the row already has a dark background (from alternating rows) with white text, preserve it.
                # Otherwise, default to black for readability on potentially lighter heatmap colors.
                row_has_dark_bg = 'background-color: #2C3E50;' in styles_df_final.loc[row_idx, 'model_name'] # Check for dark background in a representative column
                final_text_color_for_heatmap = 'color: #FFFFFF;' if row_has_dark_bg else 'color: black;' # Preserve white text or set to black

                styles_df_final.loc[row_idx, diff_col] = f"{';'.join(current_style_entry_parts)}{style}; {final_text_color_for_heatmap}"

    # Apply item separation borders (top border for new items)
    new_item_mask = (df['Item Display Name'] != df['Item Display Name'].shift(1)).fillna(False)
    if not df.empty:
        # Do not apply border to the very first row of the entire table
        if df.index[0] in new_item_mask.index and new_item_mask.iloc[0]:
            new_item_mask.loc[df.index[0]] = False 
    
    for idx in df.index:
        if new_item_mask.loc[idx]:
            for col in df.columns:
                styles_df_final.loc[idx, col] += 'border-top: 2px solid #555555;'
    
    return styles_df_final


def render_comparison_section(actuals_df: pd.DataFrame, item_descriptions_df: pd.DataFrame, get_display_name_func, load_predictions_func, results_dir_path: Path, predictions_file_name: str):
    """
    Renders the comparison section, allowing users to compare models from two different result directories.
    """
    st.header("📊 Model Comparison Across Result Directories")
    st.write("Compare the performance of models from two different experiment runs.")

    available_res_sub_dirs = [f.name for f in results_dir_path.iterdir() if f.is_dir()]
    if not available_res_sub_dirs:
        st.warning("No results folders found in the 'results' directory to compare.")
        return

    # --- Select Directories A and B ---
    st.subheader("Select Result Directories for Comparison")
    col_a, col_b = st.columns(2)
    with col_a:
        selected_folder_name_a = st.selectbox("Choose Folder A:", sorted(available_res_sub_dirs), key="folder_a")
    with col_b:
        default_idx_b = 0
        if selected_folder_name_a and selected_folder_name_a in sorted(available_res_sub_dirs):
             if len(available_res_sub_dirs) > 1 and available_res_sub_dirs[0] == selected_folder_name_a:
                default_idx_b = 1
            
        selected_folder_name_b = st.selectbox("Choose Folder B:", sorted(available_res_sub_dirs), index=default_idx_b, key="folder_b")


    predictions_df_a = pd.DataFrame()
    predictions_df_b = pd.DataFrame()

    if selected_folder_name_a:
        predictions_df_a = load_predictions_func(results_dir_path / selected_folder_name_a, predictions_file_name)
    if selected_folder_name_b:
        predictions_df_b = load_predictions_func(results_dir_path / selected_folder_name_b, predictions_file_name)

    if predictions_df_a.empty and predictions_df_b.empty:
        st.info("Please select valid result folders and ensure prediction files are present in at least one folder.")
        return
    elif predictions_df_a.empty:
        st.warning(f"No valid predictions found for Folder A: '{selected_folder_name_a}'.")
    elif predictions_df_b.empty:
        st.warning(f"No valid predictions found for Folder B: '{selected_folder_name_b}'.")
    
    # Calculate metrics for each dataset
    metrics_a = calculate_metrics(actuals_df, predictions_df_a)
    metrics_b = calculate_metrics(actuals_df, predictions_df_b)

    if metrics_a.empty and metrics_b.empty:
        st.info("No metrics could be calculated for either selected folder.")
        return
    
    # --- Merge Metrics DataFrames ---
    comparison_df = pd.merge(
        metrics_a.rename(columns={'RMSE': 'RMSE A', 'MAE': 'MAE A', 'MAPE': 'MAPE A'}),
        metrics_b.rename(columns={'RMSE': 'RMSE B', 'MAE': 'MAE B', 'MAPE': 'MAPE B'}),
        on=['item_id', 'model_name'],
        how='outer' 
    )

    comparison_df = comparison_df.fillna(value=np.nan)

    # Calculate Percentage Differences
    metrics = ['RMSE', 'MAE', 'MAPE']
    for metric in metrics:
        col_a = f'{metric} A'
        col_b = f'{metric} B'
        diff_col = f'{metric} A B Diff %'

        condition = comparison_df[col_a].notna() & comparison_df[col_b].notna()
        
        # Calculate percentage difference (A-B)/A * 100
        # If (A - B) is positive, B is better. If (A - B) is negative, A is better.
        comparison_df.loc[condition, diff_col] = ((comparison_df.loc[condition, col_a] - comparison_df.loc[condition, col_b]) / comparison_df.loc[condition, col_a]) * 100
        comparison_df.loc[comparison_df[col_a] == 0, diff_col] = np.nan # Avoid division by zero
        comparison_df.loc[comparison_df[col_a].isna() | comparison_df[col_b].isna(), diff_col] = np.nan # Set diff to NaN if either A or B is NaN

    # Add Display Name
    comparison_df['Item Display Name'] = comparison_df['item_id'].apply(lambda x: get_display_name_func(x, item_descriptions_df))

    # Sort for consistent display
    comparison_df.sort_values(by=['Item Display Name', 'model_name'], inplace=True)

    # --- Item Filter for Comparison Table ---
    st.subheader("Filter Comparison Table by Item(s)")
    all_comparison_items_original = comparison_df['item_id'].unique().tolist()
    all_comparison_items_display_formatted = []
    for item_id_original in all_comparison_items_original:
        display_name = get_display_name_func(item_id_original, item_descriptions_df)
        if display_name != item_id_original:
            all_comparison_items_display_formatted.append(f"{display_name} ({item_id_original})")
        else:
            all_comparison_items_display_formatted.append(item_id_original)
    
    all_comparison_options_with_all = ['All Items'] + sorted(all_comparison_items_display_formatted)
    
    selected_comparison_items_display = st.multiselect(
        "Choose items to display in the comparison table:",
        options=all_comparison_options_with_all,
        default=['All Items']
    )

    filtered_comparison_df = comparison_df.copy()
    if 'All Items' not in selected_comparison_items_display:
        selected_comparison_items_original = []
        for display_name_selected in selected_comparison_items_display:
            if "(" in display_name_selected and ")" in display_name_selected:
                original_id = display_name_selected.split('(')[-1].strip(')')
            else:
                original_id = display_name_selected
            selected_comparison_items_original.append(original_id)
        filtered_comparison_df = comparison_df[comparison_df['item_id'].isin(selected_comparison_items_original)].copy()

    if filtered_comparison_df.empty:
        st.info("No data to display for the selected items and comparison folders.")
        return

    # Reorder columns for final display and drop 'item_id'
    final_cols = ['Item Display Name', 'model_name'] # Removed 'item_id'
    for metric in metrics:
        final_cols.append(f'{metric} A')
        final_cols.append(f'{metric} B')
        final_cols.append(f'{metric} A B Diff %')
    
    filtered_comparison_df = filtered_comparison_df[final_cols].reset_index(drop=True)

    # Prepare DataFrame for styling (with star icons)
    # First, format all numeric columns into strings in display_df_with_icons
    display_df_with_icons = filtered_comparison_df.copy()
    numeric_df_for_heatmap = filtered_comparison_df.copy() # Keep a numeric copy for heatmap calculation

    for index, row in display_df_with_icons.iterrows():
        for metric in metrics:
            col_a = f'{metric} A'
            col_b = f'{metric} B'
            diff_col = f'{metric} A B Diff %'

            # Format RMSE/MAE/MAPE A and B
            val_a_numeric = numeric_df_for_heatmap.loc[index, col_a]
            val_b_numeric = numeric_df_for_heatmap.loc[index, col_b]
            
            display_df_with_icons.loc[index, col_a] = f"{val_a_numeric:.2f}%" if metric == 'MAPE' and pd.notna(val_a_numeric) else (f"{val_a_numeric:.2f}" if pd.notna(val_a_numeric) else "N/A")
            display_df_with_icons.loc[index, col_b] = f"{val_b_numeric:.2f}%" if metric == 'MAPE' and pd.notna(val_b_numeric) else (f"{val_b_numeric:.2f}" if pd.notna(val_b_numeric) else "N/A")

            # Format Diff %
            diff_val = numeric_df_for_heatmap.loc[index, diff_col]
            display_df_with_icons.loc[index, diff_col] = f"{diff_val:.2f}%" if pd.notna(diff_val) else "N/A"

    # Now, add star icon for the best result (lower value) for RMSE, MAE, MAPE for each item across all models
    unique_items_for_star = display_df_with_icons['Item Display Name'].unique()
    for item_display_name in unique_items_for_star:
        # Get the numeric subset for this item from the original numeric DataFrame
        item_group_numeric_for_min = numeric_df_for_heatmap[numeric_df_for_heatmap['Item Display Name'] == item_display_name]
        
        for metric in metrics:
            col_a = f'{metric} A'
            col_b = f'{metric} B'

            # Combine A and B values for finding the minimum within the item group
            combined_metric_values = pd.concat([
                item_group_numeric_for_min[col_a],
                item_group_numeric_for_min[col_b]
            ]).dropna()

            if not combined_metric_values.empty:
                min_val_for_item_metric = combined_metric_values.min()

                # Iterate through the original rows of the current item group (using its indices) to apply stars
                for original_idx in item_group_numeric_for_min.index:
                    val_a_numeric = numeric_df_for_heatmap.loc[original_idx, col_a]
                    val_b_numeric = numeric_df_for_heatmap.loc[original_idx, col_b]

                    # Apply star to the best model's value (A or B) in that row
                    if pd.notna(val_a_numeric) and val_a_numeric == min_val_for_item_metric:
                        display_df_with_icons.loc[original_idx, col_a] = f"⭐ {display_df_with_icons.loc[original_idx, col_a]}"
                    
                    if pd.notna(val_b_numeric) and val_b_numeric == min_val_for_item_metric:
                        display_df_with_icons.loc[original_idx, col_b] = f"⭐ {display_df_with_icons.loc[original_idx, col_b]}"


    # Apply styling
    styled_table = display_df_with_icons.style.apply(
        apply_comparison_styling, # Directly pass the function
        numeric_df_for_heatmap=numeric_df_for_heatmap, # Pass numeric_df_for_heatmap as a keyword argument
        axis=None
    )
    
    st.dataframe(styled_table, use_container_width=True)
