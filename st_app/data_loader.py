# st_app/data_loader.py

import streamlit as st
import pandas as pd
from pathlib import Path
import warnings

# Suppress specific warnings if needed
warnings.filterwarnings("ignore")

# Removed hardcoded file names from here. They will be passed as arguments.


@st.cache_data
def load_predictions(results_base_path: Path, predictions_file_name: str) -> pd.DataFrame:
    """
    Loads prediction data from the specified results folder.
    Args:
        results_base_path (Path): The base path to the results directory.
        predictions_file_name (str): The name of the predictions CSV file.
    """
    predictions_file = results_base_path / predictions_file_name
    if not predictions_file.exists():
        st.error(f"Error: `{predictions_file_name}` not found in the selected folder: {results_base_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(predictions_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading or parsing `{predictions_file_name}`: {e}")
        return pd.DataFrame()

@st.cache_data
def load_actuals(data_folder_path: Path, data_file_name: str) -> pd.DataFrame:
    """
    Loads actual data from the specified data folder.
    Args:
        data_folder_path (Path): The base path to the data directory.
        data_file_name (str): The name of the actuals CSV file.
    """
    actual_file = data_folder_path / data_file_name
    if not actual_file.exists():
        st.error(f"Error: {data_file_name} not found in the selected folder: {data_folder_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(actual_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df
    except Exception as e:
        st.error(f"Error loading or parsing `{data_file_name}`: {e}")
        return pd.DataFrame()

@st.cache_data
def load_item_descriptions(data_folder_path: Path, item_description_file_name: str) -> pd.DataFrame:
    """
    Loads item descriptions from the specified data folder.
    Args:
        data_folder_path (Path): The base path to the data directory.
        item_description_file_name (str): The name of the item description CSV file.
    """
    item_file = data_folder_path / item_description_file_name
    if not item_file.exists():
        st.warning(f"Warning: Item description file `{item_description_file_name}` not found at: {item_file}. Using raw item IDs.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(item_file)
        return df
    except Exception as e:
        st.error(f"Error loading or parsing `{item_description_file_name}`: {e}")
        return pd.DataFrame()

def get_display_name(item_id: str, item_descriptions_df: pd.DataFrame) -> str:
    """
    Returns the descriptive item name if available, otherwise returns the original item_id.
    This function should be used with the loaded item_descriptions_df.
    """
    if not item_descriptions_df.empty and \
       'original item name' in item_descriptions_df.columns and \
       'new item name' in item_descriptions_df.columns:
        item_name_mapping = pd.Series(
            item_descriptions_df['new item name'].values,
            index=item_descriptions_df['original item name']
        ).to_dict()
        return item_name_mapping.get(item_id, item_id)
    return item_id
