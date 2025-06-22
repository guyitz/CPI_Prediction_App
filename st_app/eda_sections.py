# st_app/eda_sections.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# This function will orchestrate the display of the entire EDA section
def render_eda_section(actuals_df: pd.DataFrame, item_descriptions_df: pd.DataFrame, get_display_name_func):
    """
    Renders the Exploratory Data Analysis (EDA) section of the Streamlit app.

    Args:
        actuals_df (pd.DataFrame): DataFrame containing actual time series data.
        item_descriptions_df (pd.DataFrame): DataFrame containing item descriptions for mapping.
        get_display_name_func (function): A function to get the display name for an item_id.
    """
    st.header("🔍 Exploratory Data Analysis")
    
    if actuals_df.empty:
        st.warning("Cannot perform EDA: Actuals data is not loaded.")
        return # Exit early if no data

    st.subheader("Select an Item for EDA")
    
    all_items_original_eda = actuals_df['item_id'].unique().tolist()
    
    # Create display names in the format "New Name (Original ID)"
    all_items_display_eda_formatted = []
    for item_id_original in all_items_original_eda:
        display_name = get_display_name_func(item_id_original, item_descriptions_df)
        if display_name != item_id_original: # If a mapping exists
            all_items_display_eda_formatted.append(f"{display_name} ({item_id_original})")
        else: # If no mapping exists, use original ID directly
            all_items_display_eda_formatted.append(item_id_original)

    selected_item_display_eda = st.selectbox(
        "Choose an item for detailed analysis:",
        options=sorted(all_items_display_eda_formatted)
    )
    
    # Extract the original item_id from the selected display name
    if "(" in selected_item_display_eda and ")" in selected_item_display_eda:
        selected_item_original_eda = selected_item_display_eda.split('(')[-1].strip(')')
    else:
        selected_item_original_eda = selected_item_display_eda # If no parentheses, it's already the original ID


    st.subheader(f"Analysis for: {get_display_name_func(selected_item_original_eda, item_descriptions_df)}") # Use display name for subheader
    
    # Filter raw data for the selected item
    item_raw_data_full = actuals_df[actuals_df['item_id'] == selected_item_original_eda].copy()
    
    # 1. Raw data of the series
    st.write("#### Raw Data:")
    # Display the dataframe (reset index for display clarity if it's currently timestamp-indexed)
    st.dataframe(item_raw_data_full.reset_index(drop=False))

    # 2. Line graph of the original data with year selection
    st.write("#### Line Graph of Original Data:")
    if not item_raw_data_full.empty:
        min_year_data = item_raw_data_full.index.min().year # Use index for min/max year
        max_year_data = item_raw_data_full.index.max().year # Use index for min/max year

        # Initialize session state for slider values if not already set
        if 'eda_start_year' not in st.session_state:
            st.session_state.eda_start_year = max(min_year_data, 2000) # Default to 2000 or data min
        if 'eda_end_year' not in st.session_state:
            st.session_state.eda_end_year = max_year_data
        
        # Ensure session state values are within current item's data range
        st.session_state.eda_start_year = max(min_year_data, st.session_state.eda_start_year)
        st.session_state.eda_end_year = min(max_year_data, st.session_state.eda_end_year)
        st.session_state.eda_end_year = max(st.session_state.eda_end_year, st.session_state.eda_start_year) # ensure end >= start


        # Use a slider for year range selection
        year_range = st.slider(
            "Select a range of years for plot and decomposition:",
            min_value=min_year_data,
            max_value=max_year_data,
            value=(st.session_state.eda_start_year, st.session_state.eda_end_year)
        )
        # Update session state if slider value changes
        st.session_state.eda_start_year = year_range[0]
        st.session_state.eda_end_year = year_range[1]
        
        # Filter data based on selected years from slider (use index for filtering)
        filtered_data_for_plots_and_decomposition = item_raw_data_full[
            (item_raw_data_full.index.year >= st.session_state.eda_start_year) &
            (item_raw_data_full.index.year <= st.session_state.eda_end_year)
        ].copy() # Ensure we're working on a copy

        if not filtered_data_for_plots_and_decomposition.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(filtered_data_for_plots_and_decomposition.index, filtered_data_for_plots_and_decomposition['target'], label='Actual Data', color='blue')
            ax.set_title(f"{get_display_name_func(selected_item_original_eda, item_descriptions_df)} - Actual Data (Years {st.session_state.eda_start_year}-{st.session_state.eda_end_year})", fontsize=16)
            ax.set_xlabel("Timestamp", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.6)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No data available for the selected year range.")
    else:
        st.info("No raw data to plot for this item.")

    # 3. Seasonal Decomposition
    st.write("#### Seasonal Decomposition:")
    # Use the already filtered data for decomposition
    series_for_decomposition = filtered_data_for_plots_and_decomposition['target'].copy()

    if not series_for_decomposition.empty:
        try:
            # Ensure 'target' column is numeric and drop NaNs
            series_for_decomposition = pd.to_numeric(series_for_decomposition, errors='coerce').dropna()

            # Resample to monthly frequency to ensure uniform time steps for decomposition
            series_for_decomposition = series_for_decomposition.resample('MS').mean().dropna()

            # Check again if enough data points after resampling and dropping NaNs
            if len(series_for_decomposition) < 2 * 12:
                st.info(f"After resampling and cleaning, not enough data points ({len(series_for_decomposition)}) for seasonal decomposition for {get_display_name_func(selected_item_original_eda, item_descriptions_df)}. Requires at least 24 data points for monthly data.")
            else:
                decomposition = seasonal_decompose(series_for_decomposition, model='additive', period=12)
                
                fig_decomposition = decomposition.plot()
                fig_decomposition.set_size_inches(12, 8)
                fig_decomposition.suptitle(f"{get_display_name_func(selected_item_original_eda, item_descriptions_df)} - Seasonal Decomposition (Years {st.session_state.eda_start_year}-{st.session_state.eda_end_year})", y=1.02) # Adjust suptitle position
                plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent suptitle overlap
                st.pyplot(fig_decomposition)
                plt.close(fig_decomposition)
        except Exception as e:
            st.error(f"Failed to perform seasonal decomposition for {get_display_name_func(selected_item_original_eda, item_descriptions_df)}: {e}")
            st.info("Ensure the data has a valid DatetimeIndex, is numeric, and has sufficient length for decomposition.")
    else:
        st.info("Not enough data points or invalid series to perform seasonal decomposition for the selected year range.")

    # 4. Correlation with other series (Heatmap)
    st.write("#### Correlation Heatmap with Other Series:")
    
    # Get all item_ids that are not the currently selected one for correlation selection
    all_other_items_original_for_corr = [
        item_id for item_id in actuals_df['item_id'].unique().tolist()
        if item_id != selected_item_original_eda
    ]

    # Create display names for correlation multiselect in the format "New Name (Original ID)"
    all_other_items_display_corr_formatted = []
    for item_id_original in all_other_items_original_for_corr:
        display_name = get_display_name_func(item_id_original, item_descriptions_df)
        if display_name != item_id_original: # If a mapping exists
            all_other_items_display_corr_formatted.append(f"{display_name} ({item_id_original})")
        else: # If no mapping exists, use original ID directly
            all_other_items_display_corr_formatted.append(item_id_original)

    # Set 'select all' as default for the correlation multiselect
    selected_items_display_corr = st.multiselect(
        "Select other items to include in the correlation heatmap (default: all):", # Updated prompt
        options=sorted(all_other_items_display_corr_formatted),
        default=sorted(all_other_items_display_corr_formatted) # Select all by default
    )

    # Convert selected display names back to original item_ids for filtering dataframes
    selected_items_original_corr = []
    for display_name_selected in selected_items_display_corr:
        if "(" in display_name_selected and ")" in display_name_selected:
            original_id = display_name_selected.split('(')[-1].strip(')')
        else:
            original_id = display_name_selected
        selected_items_original_corr.append(original_id)

    # Add the selected main item to the list for initial correlation calculation
    # This list will include the main selected item and ALL other selected items from the multiselect
    items_for_full_correlation = [selected_item_original_eda] + selected_items_original_corr

    if len(items_for_full_correlation) > 1:
        # Filter the actuals_df for these items and the selected year range
        correlation_data_full = actuals_df[actuals_df['item_id'].isin(items_for_full_correlation)].copy()
        correlation_data_full = correlation_data_full[
            (correlation_data_full.index.year >= st.session_state.eda_start_year) &
            (correlation_data_full.index.year <= st.session_state.eda_end_year)
        ]

        # Pivot the table to have item_ids as columns
        correlation_pivot_full = correlation_data_full.pivot_table(index='timestamp', columns='item_id', values='target')
        
        # Convert values to numeric and drop rows with NaNs across all relevant columns for correlation
        for col in correlation_pivot_full.columns:
            correlation_pivot_full[col] = pd.to_numeric(correlation_pivot_full[col], errors='coerce')
        correlation_pivot_full = correlation_pivot_full.dropna()

        if not correlation_pivot_full.empty and correlation_pivot_full.shape[1] > 1: # Need at least 2 series for correlation
            # Rename columns for the heatmap to be more readable (using new item names)
            # Apply mapping to the columns that exist in the correlation_pivot_full
            current_columns_to_rename_full = {col: get_display_name_func(col, item_descriptions_df) for col in correlation_pivot_full.columns}
            correlation_pivot_full.rename(columns=current_columns_to_rename_full, inplace=True)
            
            # Calculate the full correlation matrix
            corr_matrix_full = correlation_pivot_full.corr()

            # Add slider for minimum absolute correlation
            min_abs_correlation = st.slider(
                "Minimum Absolute Correlation to Display (filtered from selected items):", # Updated prompt
                min_value=0.0,
                max_value=1.0,
                value=0.0, # Default to show all
                step=0.05,
                key='min_abs_corr_slider' # Add a unique key to prevent conflicts if other sliders are added
            )

            # Determine which items to display on the heatmap based on the threshold
            # The primary selected EDA item should always be included
            primary_display_name = get_display_name_func(selected_item_original_eda, item_descriptions_df)
            items_to_display_on_heatmap = [primary_display_name]
            
            # Filter other selected items based on their absolute correlation with the primary EDA item
            for col_name in corr_matrix_full.columns:
                if col_name != primary_display_name:
                    # Ensure the primary_display_name is actually in the columns of corr_matrix_full
                    # This avoids KeyError if, for some reason, the primary item got dropped (e.g., all NaNs)
                    if primary_display_name in corr_matrix_full.index and col_name in corr_matrix_full.columns:
                        if abs(corr_matrix_full.loc[primary_display_name, col_name]) >= min_abs_correlation:
                            items_to_display_on_heatmap.append(col_name)
            
            # Ensure uniqueness and order for heatmap axes
            items_to_display_on_heatmap = sorted(list(set(items_to_display_on_heatmap)))

            if len(items_to_display_on_heatmap) > 1:
                # Slice the full correlation matrix to get only the desired items
                final_corr_matrix = corr_matrix_full.loc[items_to_display_on_heatmap, items_to_display_on_heatmap]

                # Increased figsize factor for better readability with more items
                fig_corr, ax_corr = plt.subplots(figsize=(max(8, len(items_to_display_on_heatmap)*1.0), max(6, len(items_to_display_on_heatmap)*1.0)))
                sns.heatmap(final_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
                ax_corr.set_title(f"Correlation Heatmap for Selected Series (Years {st.session_state.eda_start_year}-{st.session_state.eda_end_year}, Min Abs Corr >= {min_abs_correlation})", fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig_corr)
                plt.close(fig_corr)
            else:
                st.info(f"No other items meet the minimum absolute correlation threshold ({min_abs_correlation}) with '{primary_display_name}' in the selected range, or not enough items selected.")
        else:
            st.info("Not enough valid data or selected series to compute a meaningful correlation heatmap for the selected range.")
    else:
        st.info("Please select more than one item (including the primary item) to compute a correlation heatmap.")

    # 5. Time Series Statistics
    st.write("#### Time Series Statistics:")
    if not filtered_data_for_plots_and_decomposition.empty:
        st.write("##### Summary Statistics:")
        summary_stats = filtered_data_for_plots_and_decomposition['target'].describe()

        # Define the order and labels for the metrics
        metrics_to_display = {
            "Count": "count",
            "Mean": "mean",
            "Std Dev": "std",
            "Min": "min",
            "25th Percentile": "25%",
            "Median (50th)": "50%",
            "75th Percentile": "75%",
            "Max": "max"
        }

        # Create columns for layout
        cols = st.columns(3) # Use 3 columns as in the user's example

        col_idx = 0
        for label, key in metrics_to_display.items():
            with cols[col_idx]:
                # Check if the key exists in summary_stats to avoid KeyError
                if key in summary_stats.index:
                    st.metric(label, f"{summary_stats[key]:.2f}", help=f"Value for {label}")
            col_idx = (col_idx + 1) % 3 # Cycle through 3 columns
        
        st.write("---") # Add a separator for clarity

        st.write("##### Augmented Dickey-Fuller Test for Stationarity:")
        series_for_adf = filtered_data_for_plots_and_decomposition['target'].dropna()
        
        if len(series_for_adf) > 0:
            try:
                adf_result = adfuller(series_for_adf)
                st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
                st.write(f"**p-value:** {adf_result[1]:.4f}")
                st.write(f"**Number of Lags Used:** {adf_result[2]}")
                st.write("**Critical Values:**")
                for key, value in adf_result[4].items():
                    st.write(f"   {key}: {value:.4f}")

                if adf_result[1] <= 0.05:
                    st.success("Conclusion: The series is likely stationary (p-value <= 0.05).")
                else:
                    st.warning("Conclusion: The series is likely non-stationary (p-value > 0.05). Differencing might be needed.")
            except Exception as e:
                st.error(f"Could not perform ADF test: {e}")
                st.info("ADF test requires sufficient data points and variation. Ensure the series is not constant and has enough observations.")
        else:
            st.info("Not enough data points to perform Augmented Dickey-Fuller Test for the selected range.")
    else:
        st.info("No data available for the selected year range to compute statistics.")
