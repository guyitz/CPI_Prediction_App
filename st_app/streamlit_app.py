# st_app/streamlit_app.py

import streamlit as st
import pandas as pd
from pathlib import Path
import warnings
import matplotlib.pyplot as plt 

# Adjust the import path for config_loader assuming it's in the project root
# and st_app is a subdirectory of the project root.
import sys
# Get the parent directory of the current script (st_app)
current_dir = Path(__file__).parent
# Get the project root directory (one level up from st_app)
project_root = current_dir.parent
# Add the project root to sys.path
sys.path.append(str(project_root))

from config_loader import load_config
# Import modules from st_app
from data_loader import load_predictions, load_actuals, load_item_descriptions, get_display_name
from eda_sections import render_eda_section
from prediction_plots import render_prediction_plots_section
from prediction_scores import render_prediction_score_section
from comparison_section import render_comparison_section # NEW IMPORT

# Suppress specific warnings if needed
warnings.filterwarnings("ignore")

##########################################################################################
#########################################Config Vars######################################
##########################################################################################
config = load_config()

# Data paths from config - these should now be resolved relative to the project root
# and passed to functions that need them.
# The data_loader functions will handle their own paths based on config values
# which are relative to the project root.
DATA_FOLDER_PATH = Path(config['paths']['data'])
RESULTS_DIR_PATH = Path(config['paths']['results_base'])

# Get data file names from config or define them here if not in config
ACTUALS_FILE_NAME = config['data']['data_file_name'] # From config as requested
ITEM_DESCRIPTION_FILE = 'item_description.csv' # Assuming this is a fixed name not in config
PREDICTIONS_FILE_NAME = "all_models_predictions.csv" # Assuming this is a fixed name not in config


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="CPI Forecast Visualizer")

st.title("📊 CPI Forecast Visualization App")

st.write(
    "Explore your time series data and forecast results."
)

# --- Sidebar for Controls ---
st.sidebar.header("Navigation")
selected_section = st.sidebar.radio(
    "Go to",
    ("Exploratory Data Analysis (EDA)", "Prediction Result Plots", "Prediction Result Score", "Comparison") # Added "Comparison"
)

st.sidebar.header("Controls") # Keep controls header separate for clarity

# Load actuals data (always loaded as it's needed for all sections)
actuals_df = load_actuals(DATA_FOLDER_PATH, ACTUALS_FILE_NAME)

# Load item descriptions (always loaded for item name mapping)
item_descriptions_df = load_item_descriptions(DATA_FOLDER_PATH, ITEM_DESCRIPTION_FILE)

# Initialize predictions_df outside the if/elif blocks to make it available globally
predictions_df = pd.DataFrame()
selected_folder_name = None # Initialize selected_folder_name as well

# Folder selection for predictions is now specific to Plot and Score sections
# For Comparison, folder selection is handled inside render_comparison_section
if selected_section in ["Prediction Result Plots", "Prediction Result Score"]:
    st.sidebar.subheader("1. Select Results Folder")
    available_res_sub_dirs = [f.name for f in RESULTS_DIR_PATH.iterdir() if f.is_dir()]
    if not available_res_sub_dirs:
        st.sidebar.warning("No results folders found in the 'results' directory.")
        selected_folder_name = None
    else:
        selected_folder_name = st.sidebar.selectbox("Choose a folder:", available_res_sub_dirs, key="main_results_folder_select") # Added key

    if selected_folder_name:
        selected_folder_path = RESULTS_DIR_PATH / selected_folder_name
        predictions_df = load_predictions(selected_folder_path, PREDICTIONS_FILE_NAME)


# --- Main Content Area based on Section Selection ---
if selected_section == "Exploratory Data Analysis (EDA)":
    render_eda_section(actuals_df, item_descriptions_df, get_display_name)

elif selected_section == "Prediction Result Plots":
    render_prediction_plots_section(actuals_df, predictions_df, item_descriptions_df, get_display_name)

elif selected_section == "Prediction Result Score":
    render_prediction_score_section(predictions_df, actuals_df, item_descriptions_df, get_display_name)

elif selected_section == "Comparison": # NEW SECTION
    render_comparison_section(actuals_df, item_descriptions_df, get_display_name, load_predictions, RESULTS_DIR_PATH, PREDICTIONS_FILE_NAME)

