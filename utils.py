"""
Utility functions for data processing, model training, and visualization.
Organized into logical sections for clarity and reusability.
"""

import os
import re
from datetime import datetime, date
import pandas as pd
import numpy as np
from scipy import stats

# =====================================================
# 1. DATA LOADING FUNCTIONS
# =====================================================

def get_files_in_date_range(directory, start_date_str, end_date_str):
    """
    Filter files by date range from directory.
    
    Args:
        directory (str): Path to directory containing files
        start_date_str (str): Start date in format "YYYY-MM-DD"
        end_date_str (str): End date in format "YYYY-MM-DD"
    
    Returns:
        list: List of filenames (without .csv extension) within date range
    """
    files = os.listdir(directory)
    file_names = []
    start_date_obj = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date_obj = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    for file in sorted(files):
        match = re.match(r'(\d{4})-(\d{2})\.csv', file)
        if match:
            year, month = match.groups()
            file_date = date(int(year), int(month), 1)
            if file_date >= start_date_obj and file_date <= end_date_obj:
                file_names.append(file.replace('.csv', ''))
    
    return file_names


def load_zipcode_mapping(filepath):
    """
    Load zipcode to state mapping from CSV file.
    
    Args:
        filepath (str): Path to uszips.csv file
    
    Returns:
        dict: Dictionary mapping zipcodes to state names
    """
    try:
        uszips_data = pd.read_csv(filepath)
        state_mapping = dict(zip(uszips_data['zip'].astype(str), uszips_data['state_name']))
        print(f"Loaded zipcode to state mapping: {len(state_mapping)} zip codes")
        return state_mapping
    except Exception as e:
        print(f"Warning: Could not load uszips data: {e}")
        return {}


def load_region_mapping(filepath):
    """
    Load state to region mapping from CSV file with multiple encoding attempts.
    
    Args:
        filepath (str): Path to county_fips_master.csv file
    
    Returns:
        dict: Dictionary mapping state names to region names
    """
    region_mapping = {}
    for encoding in ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8']:
        try:
            fips_data = pd.read_csv(filepath, encoding=encoding)
            region_mapping = dict(zip(
                fips_data['state_name'].unique(),
                fips_data.groupby('state_name')['region_name'].first()
            ))
            print(f"Loaded state to region mapping: {len(region_mapping)} states")
            return region_mapping
        except Exception as e:
            continue
    
    print("Warning: Could not load region data")
    return {}


def add_infection_from_sird(df, sird_rdata_path='SIRD.RData', dataset_date_col='EndDatetime'):
    """
    Add Infection column from SIRD.RData by matching dates.

    Args:
        df (pd.DataFrame): Target dataset
        sird_rdata_path (str): Path to SIRD.RData file
        dataset_date_col (str): Date column in target dataset

    Returns:
        pd.DataFrame: Dataset with Infection column merged from SIRD I values
    """
    if dataset_date_col not in df.columns:
        print(f"Warning: '{dataset_date_col}' not found. Infection column not added.")
        return df

    if not os.path.exists(sird_rdata_path):
        print(f"Warning: {sird_rdata_path} not found. Infection column not added.")
        return df

    try:
        import pyreadr
    except Exception as exc:
        print(f"Warning: pyreadr not available ({exc}). Infection column not added.")
        return df

    try:
        rdata = pyreadr.read_r(sird_rdata_path)
        sird_df = None
        for obj in rdata.values():
            if isinstance(obj, pd.DataFrame):
                sird_df = obj
                break

        if sird_df is None:
            print(f"Warning: No data frame found in {sird_rdata_path}. Infection column not added.")
            return df

        if 'date' not in sird_df.columns or 'I' not in sird_df.columns:
            print("Warning: SIRD data frame missing required columns 'date' and 'I'.")
            return df

        sird_local = sird_df[['date', 'I']].copy()
        sird_local['date'] = pd.to_datetime(sird_local['date']).dt.normalize()
        sird_local = sird_local.rename(columns={'date': dataset_date_col, 'I': 'Infection'})

        df_local = df.copy()
        # Avoid merge suffixes (Infection_x/Infection_y) when this function is called multiple times.
        if 'Infection' in df_local.columns:
            df_local = df_local.drop(columns=['Infection'])
        df_local[dataset_date_col] = pd.to_datetime(df_local[dataset_date_col]).dt.normalize()
        df_local = df_local.merge(sird_local, on=dataset_date_col, how='left')

        matched = df_local['Infection'].notna().sum()
        print(f"Infection column added from {sird_rdata_path}: matched {matched}/{len(df_local)} rows")
        return df_local
    except Exception as exc:
        print(f"Warning: could not merge Infection from {sird_rdata_path}: {exc}")
        return df


# =====================================================
# 2. DATA CLEANING AND PREPROCESSING FUNCTIONS
# =====================================================

def load_and_clean_file(filepath, critical_cols, required_cols):
    """
    Load a data file and perform initial cleaning.
    
    Args:
        filepath (str): Path to data file
        critical_cols (list): Critical columns for dropna
        required_cols (list): Columns to keep
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    data = pd.read_csv(filepath, sep=',', low_memory=False)
    data['EndDatetime'] = pd.to_datetime(data['EndDatetime']).dt.date
    
    print(f"Processing {filepath} - Initial data shape: {data.shape}")
    
    # Filter to keep only wave >= 11
    if 'wave' in data.columns:
        data = data[data['wave'] >= 11]
        print(f"After filtering on wave 11: {data.shape}")
    else:
        print("Warning: 'wave' column not found in data")
    
    data_local = data[required_cols].dropna(subset=critical_cols)
    
    print(f"After cleaning: {data_local.shape}")
    return data_local


def map_zipcodes_to_states(df, state_mapping):
    """
    Map zipcodes to state names in dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        state_mapping (dict): Mapping from zipcode to state name
    
    Returns:
        pd.DataFrame: Dataframe with 'state' column added
    """
    if state_mapping:
        if 'zipcode' in df.columns:
            df['state'] = df['zipcode'].astype(str).map(state_mapping)
        else:
            df['state'] = df['A3'].astype(str).str[:5].map(state_mapping)
        print(f"Zip code to state mapping completed")
    else:
        print("Warning: Zip code to state mapping is empty. States not mapped.")
    
    return df


def map_states_to_regions(df, region_mapping):
    """
    Map state names to regions in dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'state' column
        region_mapping (dict): Mapping from state name to region
    
    Returns:
        pd.DataFrame: Dataframe with 'region' column added
    """
    if region_mapping:
        df['region'] = df['state'].map(region_mapping)
        print(f"State to region mapping completed")
    else:
        print("Warning: State to region mapping is empty. Regions not mapped.")
    
    return df

def recode_demographics_post_xgboost(df):
    """Apply post-XGBoost demographic recoding used by the causal pipeline."""
    df_local = df.copy()

    if 'D1' in df_local.columns:
        d1_numeric = pd.to_numeric(df_local['D1'], errors='coerce').fillna(0).astype(int)
        df_local['D1'] = d1_numeric.replace({3: 3, 4: 3, 5: 3})
        df_local = df_local[df_local['D1'] != 3]

    if 'D2' in df_local.columns:
        d2_numeric = pd.to_numeric(df_local['D2'], errors='coerce').fillna(0).astype(int)
        df_local['D2'] = d2_numeric.replace({1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3})

    if 'D8' in df_local.columns:
        d8_numeric = pd.to_numeric(df_local['D8'], errors='coerce').fillna(0).astype(int)
        df_local['D8'] = d8_numeric.replace({1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 8: 2, 6: 3, 7: 3})

    if 'D12' in df_local.columns:
        d12_numeric = pd.to_numeric(df_local['D12'], errors='coerce').fillna(0).astype(int)
        df_local['D12'] = d12_numeric.replace({1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2})

    if 'region_code' in df_local.columns:
        # Remove region_code variable
        # df_local = df_local.drop(columns=['region_code'])
        df_local = df_local[df_local['region_code'] != 0]

    return df_local

def create_numeric_encodings(state_mapping, region_mapping):
    """
    Create numeric encodings for states and regions.
    
    Args:
        state_mapping (dict): Mapping from zipcode to state name
        region_mapping (dict): Mapping from state name to region
    
    Returns:
        tuple: (state_numeric_mapping, region_numeric_mapping)
    """
    state_numeric_mapping = {
        state: idx for idx, state in enumerate(sorted(set(state_mapping.values())), 1)
    }
    region_numeric_mapping = {
        region: idx for idx, region in enumerate(sorted(set(region_mapping.values())), 1)
    }
    
    print(f"Numeric encodings created: {len(state_numeric_mapping)} states, {len(region_numeric_mapping)} regions")
    return state_numeric_mapping, region_numeric_mapping


def apply_numeric_encodings(df, state_numeric_mapping, region_numeric_mapping):
    """
    Apply numeric encodings to dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'state' and 'region' columns
        state_numeric_mapping (dict): Numeric mapping for states
        region_numeric_mapping (dict): Numeric mapping for regions
    
    Returns:
        pd.DataFrame: Dataframe with 'state_code' and 'region_code' columns
    """
    df['state_code'] = df['state'].map(state_numeric_mapping)
    df['region_code'] = df['region'].map(region_numeric_mapping)
    return df

def load_or_extract_datasets(causal_csv_path='causal_dataset.csv',
                            extract_func=None):
    """
    Load datasets from CSV if they exist, otherwise extract them.
    
    Args:
        causal_csv_path (str): Path to causal dataset CSV
        extract_func (callable): Function to call if datasets don't exist
    
    Returns:
        tuple: data_causal_model
    """
    if os.path.exists(causal_csv_path):
        print(f"Loading existing dataset from {causal_csv_path}")
        data_causal_model = pd.read_csv(causal_csv_path)
    else:
        print(f"Dataset not found. Extracting from source data...")
        if extract_func is None:
            raise ValueError("extract_func must be provided when datasets don't exist")
        data_causal_model = extract_func()
    
    return data_causal_model


def extract_and_prepare_datasets(source_directory='../../repositorios/cmu-dates',
                                start_date="2021-05-20",
                                end_date="2022-06-30",
                                zipcode_mapping_path="utils/uszips.csv",
                                region_mapping_path="utils/county_fips_master.csv"):
    """
    Complete data extraction and preparation pipeline.
    
    Args:
        source_directory (str): Directory containing source data files
        start_date (str): Start date in format "YYYY-MM-DD"
        end_date (str): End date in format "YYYY-MM-DD"
        zipcode_mapping_path (str): Path to zipcode mapping file
        region_mapping_path (str): Path to region mapping file
    
    Returns:
        pd.DataFrame: data_causal_model
    """
    # Get files in date range
    file_names = get_files_in_date_range(source_directory, start_date, end_date)
    print(f"Processing {len(file_names)} files in date range:")
    print(file_names)

    # Load mappings
    state_mapping = load_zipcode_mapping(zipcode_mapping_path)
    region_mapping = load_region_mapping(region_mapping_path)

    # Define column specifications
    critical_cols = ['A3', 'D1', 'D2', 'D8', 'D12']
    required_cols = ['EndDatetime', 'A3', 'A4', 'D1', 'D2', 'D8', 'D12', 'D9', 'V1',
                             'G1', 'I5', 'I6_1', 'I6_2', 'I6_3', 'I6_4', 'I6_5', 'I6_6', 'I6_7', 'I6_8']
    
    # Process all data files
    data_causal_model = pd.DataFrame()
    for file in file_names:
        datafile = f'{source_directory}/{file}.csv'
        causal_local = load_and_clean_file(datafile, critical_cols, required_cols)
        data_causal_model = pd.concat([data_causal_model, causal_local], ignore_index=True)

    print(f"Causal model combined shape: {data_causal_model.shape}")

    # Enrich datasets with mappings and encodings
    data_causal_model = map_zipcodes_to_states(data_causal_model, state_mapping)
    data_causal_model = map_states_to_regions(data_causal_model, region_mapping)
    state_numeric_mapping, region_numeric_mapping = create_numeric_encodings(state_mapping, region_mapping)
    data_causal_model = apply_numeric_encodings(data_causal_model, state_numeric_mapping, region_numeric_mapping)

    # Save dataset
    data_causal_model = data_causal_model.dropna(subset=['state'])
    
    # ===== FILTERING STAGE =====
    # Remove rows with I6_* == 0 (trust ratings must be 1, 2, or 3)
    # Remove rows with V1 == 0 or V1 == 3 (keep only V1 = 1 or 2)
    initial_rows = len(data_causal_model)
    
    trust_cols = [f'I6_{i}' for i in range(1, 9)]
    # Filter out rows where ANY I6_* == 0.0 or is NA
    mask_i6 = ~((data_causal_model[trust_cols] == 0.0).any(axis=1) | data_causal_model[trust_cols].isna().any(axis=1))
    data_causal_model = data_causal_model[mask_i6]

    # Filter out rows where D9 == 0.0
    mask_D9 = ~(data_causal_model["D9"] == 0.0)
    data_causal_model = data_causal_model[mask_D9]
    
    # Filter out rows where V1 == 0.0 or V1 == 3.0 (keep only V1 = 1 or 2)
    mask_v1 = ~(data_causal_model['V1'].isin([0.0, 3.0]))
    data_causal_model = data_causal_model[mask_v1]
    
    # Subtract 1 from V1 (converts 1,2 to 0,1)
    data_causal_model['V1'] = data_causal_model['V1'] - 1

    # Recode G1 to reverse order (1 -> 4, 2 -> 3, 3 -> 2, 4 -> 1)
    data_causal_model['G1'] = data_causal_model['G1'].replace({1: 4, 2: 3, 3: 2, 4: 1})
    
    final_rows = len(data_causal_model)
    removed_rows = initial_rows - final_rows
    print(f"\nData Filtering (I6_*==0 and V1==0 or 3):")
    print(f"  Before: {initial_rows:,} rows")
    print(f"  After:  {final_rows:,} rows")
    print(f"  Removed: {removed_rows:,} rows ({100*removed_rows/initial_rows:.1f}%)")
    
    data_causal_model.to_csv('causal_dataset.csv', index=False)
    data_causal_model.head(1000).to_csv('causal_dataset_sample.csv', index=False)
    print("Saved causal_dataset.csv and causal_dataset_sample.csv")

    return data_causal_model

def mean_difference(y_old: np.ndarray, y_new: np.ndarray) -> float:
    """
    Compute the difference in mean between two distributions.
    Used as a difference estimation function for causal strength calculation.
    
    Args:
        y_old: Original predictions (before intervention)
        y_new: New predictions (after intervention)
    
    Returns:
        float: Difference in mean (mean of y_new - mean of y_old)
    """
    mean_old = np.mean(y_old)
    mean_new = np.mean(y_new)
    return mean_new - mean_old

def cramers_v(x, y):
    """
    Compute Cramér's V statistic for categorical-categorical association.
    
    Args:
        x, y: 1D arrays of categorical data
    
    Returns:
        float: Cramér's V value (0 to 1)
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0
    cramers = np.sqrt(chi2 / (n * min_dim)) if n > 0 else 0
    return cramers