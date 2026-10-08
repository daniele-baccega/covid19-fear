import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from dowhy import gcm, CausalModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.ticker import PercentFormatter

from utils import *

rng = np.random.default_rng(42)

# =====================================================
# DATA EXTRACTION AND LOADING
# =====================================================

data_causal_model = load_or_extract_datasets(
    causal_csv_path='data/causal_dataset.csv',
    extract_func=extract_and_prepare_datasets
)
data_causal_model = add_infection_from_sird(data_causal_model, 'data/SIRD.RData', 'EndDatetime')

results_dir = 'results_I5_G1'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

final_dataset_path = os.path.join(results_dir, 'causal_dataset_final.csv')
if os.path.exists(final_dataset_path):
    print(f"Skipping final post-processing: existing file found at {final_dataset_path}")
    data_causal_model = pd.read_csv(final_dataset_path, parse_dates=['EndDatetime'])
else:
    data_causal_model = data_causal_model.dropna(subset=['G1'])

    i5_cols = [f'I5_{i}' for i in range(1, 10)]
    data_dummies = data_causal_model['I5'].fillna('').str.get_dummies(sep=',').add_prefix('I5_')

    for col in i5_cols:
        data_causal_model[col] = data_dummies[col].astype(float) if col in data_dummies.columns else 0.0

    data_causal_model = data_causal_model.drop(columns="I5")

    data_causal_model = add_infection_from_sird(data_causal_model, 'data/SIRD.RData', 'EndDatetime')

    for col in data_causal_model.columns:
        if col not in ['EndDatetime', 'Infection']:
            data_causal_model[col] = pd.to_numeric(data_causal_model[col], errors='coerce')

    if 'A4' in data_causal_model.columns:
        initial_rows = len(data_causal_model)
        data_causal_model = data_causal_model[data_causal_model['A4'] >= 0]
        removed_rows = initial_rows - len(data_causal_model)
        print(f"Removed {removed_rows} rows with negative A4 values")

    if 'Infection' in data_causal_model.columns:
        data_causal_model['Infection'] = pd.to_numeric(
            data_causal_model['Infection'], errors='coerce'
        ).fillna(0.0)

    data_causal_model = data_causal_model[data_causal_model['A4'] <= 100]
    data_causal_model = recode_demographics_post_xgboost(data_causal_model)

    data_causal_model['EndDatetime'] = pd.to_datetime(data_causal_model['EndDatetime'])
    data_causal_model = data_causal_model.sort_values('EndDatetime').reset_index(drop=True)

    infection_by_date = data_causal_model.groupby('EndDatetime')['Infection'].mean().sort_index()
    infection_smoothed = infection_by_date.rolling(window=60, center=True, min_periods=1).mean()
    infection_derivative = infection_smoothed.diff()

    trend_mapping = np.where(infection_derivative >= 0, 1, -1)
    trend_map_dict = dict(zip(infection_by_date.index, trend_mapping))
    trend_map_dict[infection_by_date.index[0]] = -1

    data_causal_model['infection_trend'] = (
        data_causal_model['EndDatetime'].map(trend_map_dict).astype(int)
    )

    oct_start = pd.to_datetime('2021-10-01')
    dec_end = pd.to_datetime('2021-12-31')

    infection_oct_dec = infection_by_date[
        (infection_by_date.index >= oct_start) & (infection_by_date.index <= dec_end)
    ]
    trough_date_min = infection_oct_dec.idxmin()
    trough_value = infection_oct_dec.min()
    
    # Find the first date after trough where derivative becomes positive
    dates_after_trough = infection_derivative[infection_derivative.index > trough_date_min]
    positive_deriv_mask = dates_after_trough >= 0
    if positive_deriv_mask.any():
        trough_date = dates_after_trough[positive_deriv_mask].index[0]
    else:
        trough_date = trough_date_min

    print(f"\nWave Calculation:")
    print(f"  Start date: {infection_by_date.index[0].date()}")
    print(f"  Trough (minimum in Oct-Dec 2021) at: {trough_date_min.date()}, Infection: {trough_value:.2f}")
    print(f"  Wave cutoff (first positive derivative after trough) at: {trough_date.date()}")
    print(f"  Wave 1: {infection_by_date.index[0].date()} to {trough_date.date()}")
    print(f"  Wave 2: {trough_date.date()} to {infection_by_date.index[-1].date()}")

    data_causal_model['wave'] = np.where(
        data_causal_model['EndDatetime'] <= trough_date,
        1,
        2
    )

    data_causal_model = data_causal_model.drop(columns=['state', 'region'], errors='ignore')

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    infection_by_date_plot = data_causal_model.groupby('EndDatetime')['Infection'].mean()
    infection_by_date_smoothed = infection_by_date_plot.rolling(window=60, center=True, min_periods=1).mean()
    
    # Add derivative-based background shading (positive=green, negative=red)
    infection_derivative = infection_by_date_smoothed.diff()
    
    dates_list = sorted(data_causal_model['EndDatetime'].unique())
    trend_pos_added = False
    trend_neg_added = False
    
    for i in range(len(dates_list) - 1):
        # Get trend value
        trend_mask = data_causal_model['EndDatetime'] == dates_list[i]
        if trend_mask.sum() > 0:
            trend_val = data_causal_model[trend_mask]['infection_trend'].iloc[0]
            
            # Add trend background (positive=green, negative=red)
            if trend_val > 0:
                ax.axvspan(dates_list[i], dates_list[i+1], alpha=0.08, color='green', zorder=0,
                          label='Rising/Stable' if not trend_pos_added else '')
                trend_pos_added = True
            else:
                ax.axvspan(dates_list[i], dates_list[i+1], alpha=0.08, color='red', zorder=0,
                          label='Decreasing' if not trend_neg_added else '')
                trend_neg_added = True
    
    # Plot infection curve
    ax.plot(
        infection_by_date_smoothed.index,
        infection_by_date_smoothed.values,
        linewidth=2.5,
        color='black',
        label='60-day Rolling Mean',
        zorder=3
    )
    
    # Add wave cutoff line
    ax.axvline(
        x=trough_date,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Wave Cutoff ({trough_date.date()})',
        zorder=2
    )
    
    ax.set_ylabel('Infection Count', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'infection_wave_trend_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization: {plot_path}")
    plt.close()

    initial_rows = len(data_causal_model)
    data_causal_model = data_causal_model.dropna()
    removed_rows = initial_rows - len(data_causal_model)
    print(f"\nRemoving rows with any NaN values:")
    print(f"  Before: {initial_rows:,} rows")
    print(f"  After:  {len(data_causal_model):,} rows")
    print(f"  Removed: {removed_rows:,} rows ({100*removed_rows/initial_rows:.1f}%)")

    # =====================================================
    # RESTRICTIONS DATA
    # =====================================================

    # Load data
    restrictions_all = pd.read_csv("data/OxCGRT_compact_subnational_v1.csv")

    # Convert date
    restrictions_all["Date"] = pd.to_datetime(restrictions_all["Date"], format="%Y%m%d")

    # Create filtered dataset
    restrictions_all = (
        restrictions_all
        .assign(date=restrictions_all["Date"])
        .loc[
            (restrictions_all["Date"] <= data_causal_model["EndDatetime"].max()) &
            (restrictions_all["Date"] >= data_causal_model["EndDatetime"].min()) &
            (restrictions_all["CountryName"] == "United States"),
            [
                "Date",
                "RegionName",
                "StringencyIndex_Average"
            ]
        ]
    )

    restrictions_all = restrictions_all.rename(columns={"RegionName": "state"})
    restrictions_all = restrictions_all.dropna(subset=["state"])

    zipcode_mapping_path="utils/uszips.csv"
    region_mapping_path="utils/county_fips_master.csv"
    state_mapping = load_zipcode_mapping(zipcode_mapping_path)
    region_mapping = load_region_mapping(region_mapping_path)
    restrictions_all = map_states_to_regions(restrictions_all, region_mapping)
    state_numeric_mapping, region_numeric_mapping = create_numeric_encodings(state_mapping, region_mapping)
    restrictions_all = apply_numeric_encodings(restrictions_all, state_numeric_mapping, region_numeric_mapping)
    restrictions_all = restrictions_all.groupby(['Date', 'region_code'], as_index=False)["StringencyIndex_Average"].mean()
    restrictions_all.to_csv(os.path.join(results_dir, 'restrictions_data.csv'), index=False)

    region_code_labels = {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'}
    region_pastel_colors = {
        'Northeast': '#7FBE7F',
        'Midwest': '#E07777',
        'South': '#D99957',
        'West': '#6BA3E5'
    }

    plt.figure(figsize=(14, 2))
    for region_code, group_data in restrictions_all.groupby('region_code'):
        region_name = region_code_labels.get(region_code, str(region_code))
        region_color = region_pastel_colors.get(region_name, '#BDBDBD')
        plt.plot(
            group_data['Date'],
            group_data['StringencyIndex_Average'],
            label=f"{region_name}",
            color=region_color,
            linewidth=2.0
        )
    for y in [20, 30, 40]:
        plt.axhline(y=y, color='grey', linestyle='--', linewidth=1)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylabel('Stringency Index', fontsize=9)
    plt.legend(title='Region', fontsize=9)
    plt.grid(True, alpha=0.3)
    restrictions_dist_path = os.path.join(results_dir, 'distribution_restrictions.png')
    plt.savefig(restrictions_dist_path, dpi=300, bbox_inches='tight')
    print(f"Saved restrictions distribution plot: {restrictions_dist_path}")
    plt.close()

    # Ensure merge keys have matching datetime dtype even when cached CSVs are loaded.
    data_causal_model['EndDatetime'] = pd.to_datetime(data_causal_model['EndDatetime'], errors='coerce')
    restrictions_all['Date'] = pd.to_datetime(restrictions_all['Date'], errors='coerce')

    data_causal_model = data_causal_model.merge(
        restrictions_all[['Date', 'StringencyIndex_Average']],
        left_on='EndDatetime',
        right_on='Date',
        how='left'
    ).drop(columns=['Date'])

    # Categorize StringencyIndex_Average: [0, 10) -> 1, [10, 20) -> 2, ..., [90, 100] -> 10
    data_causal_model['StringencyIndex_Average'] = pd.cut(
        data_causal_model['StringencyIndex_Average'],
        bins=np.arange(0, 110, 10),
        labels=np.arange(1, 11),
        right=False
    ).astype(float)

    data_causal_model.to_csv(final_dataset_path, index=False)
    cols = [col for col in data_causal_model.columns if col != 'G1'] + ['G1']
    data_causal_model = data_causal_model[cols]
    data_causal_model.head(n=1000).to_csv(
        os.path.join(results_dir, 'causal_dataset_final_sample.csv'),
        index=False
    )

    # Normalize each column in [0, 1]
    for col in data_causal_model.columns:
        if col not in ['EndDatetime', "G1"]:
            min_val = data_causal_model[col].min()
            max_val = data_causal_model[col].max()
            if max_val > min_val:
                data_causal_model[col] = (data_causal_model[col] - min_val) / (max_val - min_val)
            else:
                data_causal_model[col] = 0.0

    # Convert G1 column to integer for causal inference
    data_causal_model['G1'] = data_causal_model['G1'].astype(int)
    
    # Remove EndDatetime, state_code, Infection and save the final dataset for causal inference (both entire and sample)
    causal_inference_dataset = data_causal_model.drop(columns=['EndDatetime', 'state_code', 'Infection'], errors='ignore')
    causal_inference_dataset.to_csv(os.path.join(results_dir, 'causal_dataset_final_for_inference.csv'), index=False)
    causal_inference_dataset.head(n=1000).to_csv(os.path.join(results_dir, 'causal_dataset_final_for_inference_sample.csv'), index=False)



# =====================================================
# ADD INFECTION TREND AND WAVE
# =====================================================

print("\n" + "="*60)
print("Infection Trend and Wave Variables")
print("="*60)

if 'infection_trend' in data_causal_model.columns:
    trend_mapping = {-1: 'decreasing', 1: 'rising_or_stable'}
    data_causal_model['infection_trend_cat'] = data_causal_model['infection_trend'].map(trend_mapping)

    print(f"Infection Trend Distribution (based on derivative):")
    print(data_causal_model['infection_trend_cat'].value_counts())
    print(f"\nTrend values: 1 (rising or stable - derivative >= 0), -1 (decreasing - derivative < 0)")

if 'wave' in data_causal_model.columns:
    print(f"\nWave Distribution:")
    print(data_causal_model['wave'].value_counts().sort_index())

# =====================================================
# DISTRIBUTION PLOTS (APPROPRIATE FOR DATA TYPES)
# =====================================================

print("\n" + "="*60)
print("Variable Distribution Analysis")
print("="*60)

source_cols = ["I5_1", "I5_2", "I5_3", "I5_4", "I5_5", "I5_6", "I5_7", "I5_8", "I5_9"]
trust_cols = ["I6_1", "I6_2", "I6_3", "I6_4", "I6_5", "I6_6", "I6_7", "I6_8"]
demographic_cols = ["D1", "D2", "D8", "D9", "D12", "region_code"]

# Define label mappings
label_mappings = {
    'I5': {0: 'Not use', 1: 'Use'},
    'I6': {1: 'Not trust', 2: 'Somewhat trust', 3: 'Trust'},
    'D1': {1: 'Male', 2: 'Female'},
    'D2': {1: '18-34', 2: '35-64', 3: '65+'},
    'D8': {1: 'High school or lower', 2: 'Bachelor/Master', 3: 'Postgraduate'},
    'D9': {1: 'Yes', 2: 'No'},
    'D12': {1: 'English', 2: 'Other'},
    'region_code': {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'},
    'V1': {0: 'Yes', 1: 'No'},
    'G1': {1: 'Not at all', 2: 'A little', 3: 'Moderate', 4: 'A great deal'}
}

# Variable display names
var_names = {'I5': 'Source', 'I6': 'Trust', 'D1': 'Gender', 'D2': 'Age', 'D8': 'Education', 'D9': 'Occupation', 'D12': 'Language', 'region_code': 'Region', 'V1': 'Vaccination', 'G1': 'Fear', 'A4': 'Sick'}

# Specific labels for I5 and I6 information sources
i5_labels = {
    'I5_1': 'Doctors (Source)',
    'I5_2': 'Scientists (Source)',
    'I5_3': 'CDC (Source)',
    'I5_4': 'Government (Source)',
    'I5_5': 'Politicians (Source)',
    'I5_6': 'Journalists (Source)',
    'I5_7': 'Family and friends (Source)',
    'I5_8': 'Religious leaders (Source)',
    'I5_9': 'None of the above (Source)'
}

i6_labels = {
    'I6_1': 'Doctors',
    'I6_2': 'Scientists',
    'I6_3': 'CDC',
    'I6_4': 'Government',
    'I6_5': 'Politicians',
    'I6_6': 'Journalists',
    'I6_7': 'Family and friends',
    'I6_8': 'Religious leaders'
}

# I5 (Binary Information Sources) - Stacked bar plot
if all(col in data_causal_model.columns for col in source_cols):
    # Prepare data for stacked bar plot
    i5_data = []
    for col in source_cols:
        value_counts = data_causal_model[col].value_counts().sort_index()
        percentages = (value_counts / value_counts.sum()) * 100
        i5_data.append(percentages)
    
    # Create DataFrame for easier stacking
    i5_df = pd.DataFrame(i5_data, index=[i5_labels[col] for col in source_cols])
    i5_df = i5_df.fillna(0)
    # Reverse column order so "Use" (1) is at bottom
    i5_df = i5_df[[col for col in sorted(i5_df.columns, reverse=True)]]
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    i5_df.plot(kind='bar', stacked=True, ax=ax, color=['#B3E5B3', '#FFB3B3'], edgecolor='black', width=0.7)
    
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_xticklabels(i5_df.index, rotation=30, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_title('Source', fontsize=16, fontweight='bold')
    ax.legend([label_mappings['I5'].get(v, str(v)) for v in i5_df.columns], title='Usage', loc='upper right', fontsize=12, title_fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    i5_dist_path = os.path.join(results_dir, 'distribution_i5_sources.png')
    plt.savefig(i5_dist_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved I5 distribution plot: {i5_dist_path}")
    plt.close()
    
    # Save I5 distribution data as CSV
    i5_csv_path = os.path.join(results_dir, 'distribution_i5_sources.csv')
    i5_df.to_csv(i5_csv_path)
    print(f"Saved I5 distribution data: {i5_csv_path}")

# I6 (Trust with 3 values) - Stacked bar plot
if all(col in data_causal_model.columns for col in trust_cols):
    # Prepare data for stacked bar plot
    i6_data = []
    for col in trust_cols:
        value_counts = data_causal_model[col].value_counts().sort_index()
        percentages = (value_counts / value_counts.sum()) * 100
        i6_data.append(percentages)
    
    # Create DataFrame for easier stacking
    i6_df = pd.DataFrame(i6_data, index=[i6_labels[col] for col in trust_cols])
    i6_df = i6_df.fillna(0)
    # Reorder columns: 3 (Trust), 2 (Somewhat), 1 (Not trust) so Trust is at bottom
    i6_df = i6_df[[col for col in [3, 2, 1] if col in i6_df.columns]]
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    i6_df.plot(kind='bar', stacked=True, ax=ax, color=['#B3E5B3', '#FFFFCC', '#FFB3B3'], edgecolor='black', width=0.7)
    
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_xticklabels(i6_df.index, rotation=30, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_title('Trust', fontsize=16, fontweight='bold')
    ax.legend(['Trust', 'Somewhat', 'Not trust'], title='Trust Level', loc='upper right', fontsize=12, title_fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    i6_dist_path = os.path.join(results_dir, 'distribution_i6_trust.png')
    plt.savefig(i6_dist_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved I6 distribution plot: {i6_dist_path}")
    plt.close()
    
    # Save I6 distribution data as CSV
    i6_csv_path = os.path.join(results_dir, 'distribution_i6_trust.csv')
    i6_df.to_csv(i6_csv_path)
    print(f"Saved I6 distribution data: {i6_csv_path}")

# Demographics (Binary/Categorical) - Barplots with %
demo_cols_exist = [col for col in demographic_cols if col in data_causal_model.columns]
if demo_cols_exist:
    n_demos = len(demo_cols_exist)
    n_cols = 6
    n_rows = (n_demos + n_cols - 1) // n_cols  # Calculate rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 4))
    for ax in axes:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    axes = axes.flatten()  # Flatten to 1D for easier iteration
    
    for idx, col in enumerate(demo_cols_exist):
        value_counts = data_causal_model[col].value_counts().sort_index()
        percentages = (value_counts / value_counts.sum()) * 100
        axes[idx].set_title(var_names.get(col, col), fontsize=22, fontweight='bold')
        axes[idx].bar(percentages.index, percentages.values, color='#D9F0D9', edgecolor='black')
        # Only show y-axis label on first plot of each row
        if idx % n_cols == 0:
            axes[idx].set_ylabel('Percentage (%)', fontsize=22)
        axes[idx].set_xticks(sorted(percentages.index))
        if col == 'D8':
            # Split Education labels on 2 rows
            d8_labels = [label_mappings[col].get(v, str(v)) for v in sorted(percentages.index)]
            d8_labels_wrapped = [label.replace('/', '/\n').replace(' or ', '\nor ') for label in d8_labels]
            axes[idx].set_xticklabels(d8_labels_wrapped, fontsize=15)
        else:
            axes[idx].set_xticklabels([label_mappings[col].get(v, str(v)) for v in sorted(percentages.index)], fontsize=15)            
        axes[idx].tick_params(axis='y', labelsize=14)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Hide any unused subplots
    for idx in range(len(demo_cols_exist), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    demo_dist_path = os.path.join(results_dir, 'distribution_demographics.png')
    plt.savefig(demo_dist_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved demographics distribution plot: {demo_dist_path}")
    plt.close()
    
    # Save demographics distribution data as CSV
    demo_data_all = []
    for col in demo_cols_exist:
        value_counts = data_causal_model[col].value_counts().sort_index()
        percentages = (value_counts / value_counts.sum()) * 100
        for val, pct in percentages.items():
            label = label_mappings[col].get(val, str(val))
            demo_data_all.append({
                'Variable': var_names.get(col, col),
                'Value': label,
                'Percentage': pct,
                'Count': value_counts[val]
            })
    demo_csv_df = pd.DataFrame(demo_data_all)
    demo_csv_path = os.path.join(results_dir, 'distribution_demographics.csv')
    demo_csv_df.to_csv(demo_csv_path, index=False)
    print(f"Saved demographics distribution data: {demo_csv_path}")

# G1 (Fear - 4 values), V1 (Binary), A4 (0-100 continuous-like)
fig, axes = plt.subplots(1, 3, figsize=(14, 3))
for ax in axes:
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
fig.patch.set_alpha(0)

# G1 (4 values)
if 'G1' in data_causal_model.columns:
    g1_counts = data_causal_model['G1'].value_counts().sort_index()
    g1_pct = (g1_counts / g1_counts.sum()) * 100
    axes[0].set_title('Fear', fontsize=12, fontweight='bold')
    axes[0].bar(g1_pct.index, g1_pct.values, color='#E6D9FF', edgecolor='black')
    axes[0].set_ylabel('Percentage (%)', fontsize=13)
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_xticklabels(['Not at all', 'A little', 'Moderate', 'Great deal'], fontsize=13)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')

# V1 (Binary - note: already converted to 0,1)
if 'V1' in data_causal_model.columns:
    v1_counts = data_causal_model['V1'].value_counts().sort_index()
    v1_pct = (v1_counts / v1_counts.sum()) * 100
    axes[1].set_title('Vaccinated', fontsize=12, fontweight='bold')
    axes[1].bar(v1_pct.index, v1_pct.values, color='#FAC898', edgecolor='black')
    axes[1].set_ylabel('Percentage (%)', fontsize=13)
    axes[1].set_xticks(sorted(v1_pct.index))
    axes[1].set_xticklabels([label_mappings['V1'].get(v, str(v)) for v in sorted(v1_pct.index)], fontsize=13)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')

# A4 (Continuous 0-100) - Histogram with %
if 'A4' in data_causal_model.columns:
    # Use log scale for A4
    axes[2].set_title('#Sick', fontsize=12, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].hist(data_causal_model['A4'], bins=100, color='#B3D9FF', edgecolor='black', alpha=0.8, weights=np.ones(len(data_causal_model))/len(data_causal_model)*100)
    axes[2].set_ylabel('Percentage (%) in log scale', fontsize=13)
    axes[2].tick_params(axis='both', labelsize=12)
    axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
outcome_dist_path = os.path.join(results_dir, 'distribution_outcomes.png')
plt.savefig(outcome_dist_path, dpi=300, bbox_inches='tight')
print(f"Saved outcomes distribution plot: {outcome_dist_path}")
plt.close()

# Save outcomes distribution data as CSV
outcome_data_all = []
if 'G1' in data_causal_model.columns:
    g1_counts = data_causal_model['G1'].value_counts().sort_index()
    g1_pct = (g1_counts / g1_counts.sum()) * 100
    g1_labels = ['Not at all', 'A little', 'Moderate', 'Great deal']
    for val, pct in g1_pct.items():
        val_int = int(val)
        outcome_data_all.append({
            'Variable': 'Fear',
            'Value': g1_labels[val_int - 1] if 1 <= val_int <= 4 else str(val),
            'Percentage': pct,
            'Count': g1_counts[val]
        })
if 'V1' in data_causal_model.columns:
    v1_counts = data_causal_model['V1'].value_counts().sort_index()
    v1_pct = (v1_counts / v1_counts.sum()) * 100
    for val, pct in v1_pct.items():
        label = label_mappings['V1'].get(val, str(val))
        outcome_data_all.append({
            'Variable': 'Vaccinated',
            'Value': label,
            'Percentage': pct,
            'Count': v1_counts[val]
        })
outcome_csv_df = pd.DataFrame(outcome_data_all)
outcome_csv_path = os.path.join(results_dir, 'distribution_outcomes.csv')
outcome_csv_df.to_csv(outcome_csv_path, index=False)
print(f"Saved outcomes distribution data: {outcome_csv_path}")

print("\nDistribution plots and data files generated successfully.")

# =====================================================
# CAUSAL GRAPH
# =====================================================

wave_and_infection_trend_variable = ["infection_trend", "wave"]
background_col = ["D1", "D2", "D8", "D9", "D12", "region_code"] + wave_and_infection_trend_variable
trust_cols = ["I6_1", "I6_2", "I6_3", "I6_4", "I6_5", "I6_6", "I6_7", "I6_8"]
source_cols = ["I5_1", "I5_2", "I5_3", "I5_4", "I5_5", "I5_6", "I5_7", "I5_8", "I5_9"]

edges = [(bg_col, trust_col) for bg_col in background_col for trust_col in trust_cols]
edges += [(bg_col, source_col) for bg_col in background_col for source_col in source_cols]
edges += [(bg_col, "G1") for bg_col in background_col]
edges += [(bg_col, "V1") for bg_col in background_col]
edges += [(bg_col, "A4") for bg_col in ["D2", "D12", "region_code"] + wave_and_infection_trend_variable]

edges += [(trust_col, source_col) for trust_col, source_col in zip(trust_cols, source_cols)]
edges += [(source_col, "G1") for source_col in source_cols]
edges += [("V1", "G1"), ("A4", "G1")]

edges += [("region_code", "StringencyIndex_Average"),
          ("wave", "StringencyIndex_Average"),
          ("infection_trend", "StringencyIndex_Average")]

edges += [("StringencyIndex_Average", trust_col) for trust_col in trust_cols]
edges += [("StringencyIndex_Average", source_col) for source_col in source_cols]

edges += [("StringencyIndex_Average", "A4"),
          ("StringencyIndex_Average", "G1")]

print(edges)

causal_graph_all = nx.DiGraph(edges)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(causal_graph_all, seed=42)
nx.draw(
    causal_graph_all,
    pos,
    with_labels=True,
    node_size=2000,
    node_color='lightblue',
    font_size=10,
    font_weight='bold',
    arrowsize=20
)
plt.title("Causal Graph for COVID-19 Fear and Trust Analysis", fontsize=16)
plt.axis('off')
plt.savefig(os.path.join(results_dir, 'causal_graph.png'))
plt.close()

# =====================================================
# CATEGORICAL ASSOCIATION ANALYSIS
# =====================================================

print("\n" + "="*80)
print("CATEGORICAL ASSOCIATIONS - FULL DATASET (Cramér's V)")
print("Computing Cramér's V for ALL causal edges")
print("="*80)

corr_data = data_causal_model.copy()

# Extract all edges from the causal graph
all_causal_edges = list(causal_graph_all.edges())
print(f"\nTotal causal edges in graph: {len(all_causal_edges)}")

cramers_results_full = []
edges_computed = 0
edges_skipped = 0

for var1, var2 in all_causal_edges:
    if var1 in corr_data.columns and var2 in corr_data.columns:
        try:
            v_stat = cramers_v(corr_data[var1], corr_data[var2])
            cramers_results_full.append({
                'Variable 1': var1,
                'Variable 2': var2,
                'Arrow': f"{var1} -> {var2}",
                "Cramers_V": v_stat,
                'N': len(corr_data)
            })
            edges_computed += 1
        except Exception as e:
            print(f"Error computing Cramér's V for {var1} -> {var2}: {str(e)}")
            edges_skipped += 1
    else:
        edges_skipped += 1

print(f"Edges computed: {edges_computed}, Edges skipped: {edges_skipped}")

cramers_full_df = pd.DataFrame(cramers_results_full).sort_values('Cramers_V', ascending=False)
print("\nCramér's V for ALL causal edges (sorted by strength):\n")
print(cramers_full_df.to_string(index=False))
cramers_full_df.to_csv(
    os.path.join(results_dir, 'categorical_associations_cramers_v_full.csv'),
    index=False
)
print(f"\nSaved results to: {os.path.join(results_dir, 'categorical_associations_cramers_v_full.csv')}")

# =====================================================
# CAUSAL INFERENCE
# =====================================================

# =====================================================
# ATE ESTIMATION
# =====================================================

print("\n" + "="*80)
print("AVERAGE TREATMENT EFFECT (ATE) ESTIMATION")
print("Estimating causal effects of I5 sources on Fear (G1)")
print("="*80)

ate_results_path = os.path.join(results_dir, 'ate_estimation_results.csv')
ate_clustered_path = os.path.join(results_dir, 'ate_estimation_results_clustered.csv')
ate_top2_summary_path = os.path.join(results_dir, 'ate_top2_k_silhouette_scores.csv')


def print_ate_summary(ate_df: pd.DataFrame):
    print(f"\n{'='*80}\nATE Results\n{'='*80}")
    cols_to_print = [col for col in ['Treatment', 'ATE', 'Category', 'Method'] if col in ate_df.columns]
    if cols_to_print:
        print(ate_df[cols_to_print].to_string(index=False))

    if 'Category' in ate_df.columns and 'ATE' in ate_df.columns:
        print(f"\n{'='*80}\nSummary by Category\n{'='*80}")
        for category in ['Demographic', 'Information Source (I5)', 'Trust (I6)', 'Other']:
            cat_data = ate_df[ate_df['Category'] == category]
            if len(cat_data) > 0:
                print(f"\n{category} (n={len(cat_data)}): Mean ATE = {cat_data['ATE'].mean():.6f}")
                top_3 = cat_data.sort_values('ATE', key=abs, ascending=False).head(3)
                for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                    method_name = row['Method'] if 'Method' in row.index else 'NA'
                    print(f"  {idx}. {row['Treatment']:15s}: {row['ATE']:+.6f} ({method_name})")
        print(f"\n{'='*80}")


def run_ate_clustering_and_plots(ate_df: pd.DataFrame):
    if 'ATE' not in ate_df.columns or 'Treatment' not in ate_df.columns:
        print("ATE CSV missing required columns for clustering/plots. Skipping plotting.")
        return

    if len(ate_df) < 3:
        print("Not enough rows for clustering (need at least 3). Skipping clustering/plots.")
        return

    print(f"\n{'='*80}\nK-MEANS CLUSTERING OF ATE VALUES\n{'='*80}")

    ate_values = ate_df[['ATE']].values
    scaler = StandardScaler()
    ate_scaled = scaler.fit_transform(ate_values)

    max_k = min(4, len(ate_df) - 1)
    if max_k < 2:
        print("Not enough rows to evaluate k-means. Skipping clustering/plots.")
        return

    k_range = range(2, max_k + 1)
    silhouette_scores = []
    kmeans_models = {}

    print(f"\nEvaluating k-means for k = 2 to {max_k}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(ate_scaled)
        sil_score = silhouette_score(ate_scaled, cluster_labels)
        silhouette_scores.append(sil_score)
        kmeans_models[k] = kmeans
        print(f"  k={k}: Silhouette Score = {sil_score:.4f}")

    optimal_k = list(k_range)[int(np.argmax(silhouette_scores))]
    optimal_score = max(silhouette_scores)
    print(f"\nOptimal number of clusters: k={optimal_k} (Silhouette Score: {optimal_score:.4f})")

    k_to_score = dict(zip(k_range, silhouette_scores))
    top_2_k = sorted(k_to_score, key=k_to_score.get, reverse=True)[:2]
    print(f"Top-2 k by silhouette score: {top_2_k}")

    top2_summary_rows = []
    ate_df = ate_df.copy()

    for rank, k in enumerate(top_2_k, 1):
        cluster_col = f'Cluster_k{k}'
        ate_df[cluster_col] = kmeans_models[k].predict(ate_scaled)

        clustered_k_path = os.path.join(results_dir, f'ate_estimation_results_clustered_k{k}.csv')
        ate_df.to_csv(clustered_k_path, index=False)
        print(f"Saved clustered ATE results for top-{rank} k={k}: {clustered_k_path}")

        top2_summary_rows.append({
            'Rank': rank,
            'k': k,
            'Silhouette_Score': k_to_score[k],
            'Output_File': os.path.basename(clustered_k_path)
        })

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ate_df_sorted = ate_df[~ate_df['Treatment'].isin(['V1', 'A4'])].sort_values('ATE', ascending=False).reset_index(drop=True)

    # Diverging pastel mapping: min -> pastel red, 0 -> white, max -> pastel blue.
    ate_min = ate_df_sorted['ATE'].min()
    ate_max = ate_df_sorted['ATE'].max()
    if ate_min == ate_max:
        colors = ['#FFFFFF'] * len(ate_df_sorted)
    else:
        neg_range = abs(ate_min) if ate_min < 0 else 0.0
        pos_range = ate_max if ate_max > 0 else 0.0

        def _interpolate_hex(color1, color2, t):
            c1 = np.array([int(color1[i:i+2], 16) for i in (1, 3, 5)], dtype=float)
            c2 = np.array([int(color2[i:i+2], 16) for i in (1, 3, 5)], dtype=float)
            c = np.round(c1 + (c2 - c1) * t).astype(int)
            return '#{:02x}{:02x}{:02x}'.format(*c)

        pastel_red = '#f6c1c1'
        white = '#ffffff'
        pastel_blue = '#bcd8f2'

        colors = []
        for val in ate_df_sorted['ATE']:
            if val < 0 and neg_range > 0:
                # val=-neg_range -> red, val=0 -> white
                t = (val + neg_range) / neg_range
                colors.append(_interpolate_hex(pastel_red, white, t))
            elif val > 0 and pos_range > 0:
                # val=0 -> white, val=pos_range -> blue
                t = val / pos_range
                colors.append(_interpolate_hex(white, pastel_blue, t))
            else:
                colors.append(white)

    ax.bar(ate_df_sorted['Treatment'], ate_df_sorted['ATE'], color=colors, edgecolor='black')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Treatment Variable', fontsize=12)
    ax.set_ylabel('ATE', fontsize=12)
    ax.set_title(f'Average Treatment Effect (ATE) Estimates', fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(ate_df_sorted)))
    ax.set_xticklabels([i5_labels.get(ate_df_sorted['Treatment'].iloc[i], ate_df_sorted['Treatment'].iloc[i]) for i in range(len(ate_df_sorted))], rotation=30, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    ate_barplot_path = os.path.join(results_dir, f'ate_estimation_barplot.png')
    plt.savefig(ate_barplot_path, dpi=300, bbox_inches='tight')
    print(f"Saved ATE bar plot: {ate_barplot_path}")
    plt.close()

    pd.DataFrame(top2_summary_rows).to_csv(ate_top2_summary_path, index=False)
    print(f"Saved top-2 k silhouette summary: {ate_top2_summary_path}")

    final_kmeans = kmeans_models[optimal_k]
    ate_df['Cluster'] = final_kmeans.predict(ate_scaled)

    print(f"\nCluster Distribution:")
    for cluster_id in sorted(ate_df['Cluster'].unique()):
        cluster_data = ate_df[ate_df['Cluster'] == cluster_id]
        print(f"  Cluster {cluster_id}: n={len(cluster_data)}, Mean ATE={cluster_data['ATE'].mean():.6f}, Std={cluster_data['ATE'].std():.6f}")

    ate_df.to_csv(ate_clustered_path, index=False)
    print(f"\nSaved clustered ATE results: {ate_clustered_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(k_range), silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('K-Means Silhouette Score Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    silhouette_path = os.path.join(results_dir, 'ate_silhouette_scores.png')
    plt.savefig(silhouette_path, dpi=300, bbox_inches='tight')
    print(f"Saved silhouette score plot: {silhouette_path}")
    plt.close()

    print(f"\n{'='*80}")


ate_df = None

if os.path.exists(ate_results_path):
    print(f"\nSkipping ATE computation: existing file found at {ate_results_path}")
    ate_df = pd.read_csv(ate_results_path)
else:
    ate_data = data_causal_model.copy()
    ate_results = []

    print("\n" + "-"*80)
    print("ATE for All Variables → Fear (G1)")
    print("Binary treatments: propensity score | Multi-valued: linear regression")
    print("-"*80)

    binary_vars = source_cols
    # all_treatment_vars = binary_vars + ["V1", "A4"]
    all_treatment_vars = binary_vars
    all_confounders = ["D1", "D2", "D8", "D9", "D12", "region_code", "wave", "infection_trend"]

    for treatment_var in all_treatment_vars:
        if treatment_var not in ate_data.columns:
            continue

        is_binary = treatment_var in binary_vars

        try:
            model = CausalModel(
                data=ate_data,
                treatment=treatment_var,
                outcome='G1',
                common_causes=all_confounders,
                graph=causal_graph_all
            )

            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            method = "backdoor.propensity_score_stratification" if is_binary else "backdoor.linear_regression"
            ate_estimate = model.estimate_effect(identified_estimand, method_name=method)

            if treatment_var in source_cols:
                category = "Information Source (I5)"
            elif treatment_var in trust_cols:
                category = "Trust (I6)"
            else:
                category = "Other"

            method_label = "PS" if is_binary else "LR"
            print(f"{treatment_var:15s} ({category:20s}): ATE = {ate_estimate.value:+.6f} [{method_label}]")

            ate_results.append({
                'Treatment': treatment_var,
                'ATE': ate_estimate.value,
                'Method': method.split('.')[-1],
                'Category': category,
                'N_Confounders': len(all_confounders)
            })

        except Exception as e:
            print(f"{treatment_var:15s}: Error - {str(e)[:80]}")

    if ate_results:
        ate_df = pd.DataFrame(ate_results).sort_values('ATE', key=abs, ascending=False)
        ate_df.to_csv(ate_results_path, index=False)
        print(f"\nATE results saved to: {ate_results_path}")

if ate_df is not None and not ate_df.empty:
    print_ate_summary(ate_df)
    run_ate_clustering_and_plots(ate_df)
elif ate_df is not None and ate_df.empty:
    print("ATE file exists but is empty. Skipping ATE summary/plots.")

# =====================================================
# ESTIMATE OF ARROW STRENGTH
# =====================================================

print("\n" + "="*80)
print("CAUSAL INFERENCE - FULL DATASET ANALYSIS")
print("Using built graph and all available rows")
print("="*80)

strength_output_path = os.path.join(
    results_dir,
    "causal_inference_full_dataset_arrow_strength.csv"
)

def print_and_plot_arrow_strength(strength_df: pd.DataFrame):
    plot_df = strength_df.sort_values('Variance_Explained', ascending=False)

    def _pretty_node_name(node: str) -> str:
        if node in i5_labels:
            return i5_labels[node]
        if node in i6_labels:
            return f"Trust in {i6_labels[node]}"

        node_map = {
            'G1': 'Fear',
            'V1': 'Vaccinated',
            'A4': '#Sick',
            'infection_trend': 'Trend',
            'wave': 'Wave',
            'D1': 'Gender',
            'D2': 'Age',
            'D8': 'Education',
            'D9': 'Occupation',
            'D12': 'Language',
            'region_code': 'Region',
            'StringencyIndex_Average': 'Restrictions'
        }
        return node_map.get(node, node)

    def _pretty_arrow_label(arrow: str) -> str:
        if not isinstance(arrow, str) or '->' not in arrow:
            return str(arrow)
        left, right = [part.strip() for part in arrow.split('->', 1)]
        return f"{_pretty_node_name(left)}"

    # Additional vertical bar plot with CI and percentage axis.
    if all(col in plot_df.columns for col in ['VE_CI_95_Lower', 'VE_CI_95_Upper']):
        plot_df_bar = plot_df.sort_values('Variance_Explained', ascending=False).reset_index(drop=True)
        if 'Arrow' in plot_df_bar.columns:
            plot_df_bar['Arrow_Label'] = plot_df_bar['Arrow'].apply(_pretty_arrow_label)
        else:
            plot_df_bar['Arrow_Label'] = [str(i) for i in range(len(plot_df_bar))]

        x = np.arange(len(plot_df_bar))
        y = plot_df_bar['Variance_Explained'].values

        lower_err = np.maximum(y - plot_df_bar['VE_CI_95_Lower'].values, 0)
        upper_err = np.maximum(plot_df_bar['VE_CI_95_Upper'].values - y, 0)
        yerr = np.vstack([lower_err, upper_err])
        
        fig, ax1 = plt.subplots(figsize=(16, 6))
        ax1.set_facecolor("none")
        fig.patch.set_alpha(0)
        ax1.patch.set_alpha(0)
        ax1.bar(
            x,
            y,
            color='#8EC9FF',
            edgecolor='black',
            yerr=yerr,
            capsize=4,
            error_kw={'elinewidth': 1.3, 'ecolor': '#333333'}
        )
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax1.set_ylabel('Variation in Fear variance', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(plot_df_bar['Arrow_Label'], rotation=30, ha='right', fontsize=9)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        ax1.grid(True, alpha=0.3, axis='y')

        # In the second axis, I want to show % with respect to the sum of all values
        total_variance_explained = y.sum()
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax2.tick_params(axis='y', labelsize=12)
        ax2.set_ylabel('Variation in Fear variance (%)', fontsize=12)
        ax2.set_yticks(ax1.get_yticks())
        ax2.set_yticklabels([f"{(tick/total_variance_explained*100 if total_variance_explained > 0 else 0):.1f}%" for tick in ax1.get_yticks()])

        plt.tight_layout()
        arrow_barplot_path = os.path.join(results_dir, 'causal_inference_arrow_strength_bar_ci.png')
        plt.savefig(arrow_barplot_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Saved arrow strength bar plot with CI: {arrow_barplot_path}")
        plt.close()


if os.path.exists(strength_output_path):
    print(f"\nSkipping arrow strength computation: existing files found in {results_dir}")
    strength_df = pd.read_csv(strength_output_path)
    print_and_plot_arrow_strength(strength_df)
else:
    data = data_causal_model.copy()
    print(f"\nRows available: {len(data)}")

    if data.empty:
        print("No rows found in dataset. Skipping causal inference section.")
    else:
        gcm_data = data.copy()

        for node in causal_graph_all.nodes:
            if node not in gcm_data.columns:
                gcm_data[node] = 0

        gcm_columns = list(causal_graph_all.nodes)
        gcm_data = gcm_data[gcm_columns].copy()

        for node in gcm_columns:
            gcm_data[node] = pd.to_numeric(gcm_data[node], errors='coerce')

        initial_rows = len(gcm_data)
        gcm_data = gcm_data.dropna()
        removed_rows = initial_rows - len(gcm_data)
        if removed_rows > 0:
            print(f"\nRemoved {removed_rows} rows with NaN values for GCM analysis")

        print(f"GCM data shape: {gcm_data.shape} ({len(gcm_data)} rows × {len(gcm_data.columns)} columns)")

        target_nodes = ['G1']

        print("\n" + "="*80)
        print("Fitting Causal Model")
        print("="*80)

        model_save_path = os.path.join(results_dir, 'gcm_model_fitted.pkl')

        if os.path.exists(model_save_path):
            print(f"\nLoading existing fitted model from: {model_save_path}")
            try:
                with open(model_save_path, 'rb') as f:
                    gcm_model = pickle.load(f)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Refitting model instead...")
                gcm_model = None
        else:
            gcm_model = None

        if gcm_model is None:
            print(f"\nFitting new causal model...")
            gcm.util.general.set_random_seed(42)
            gcm_model = gcm.ProbabilisticCausalModel(causal_graph_all)
            gcm.auto.assign_causal_mechanisms(gcm_model, gcm_data)
            gcm.fit(gcm_model, gcm_data)

            try:
                with open(model_save_path, 'wb') as f:
                    pickle.dump(gcm_model, f)
                print(f"Model saved to: {model_save_path}")
            except Exception as e:
                print(f"Warning: Could not save model: {str(e)}")

        print(f"\nModel fitted successfully.")

        print("\n" + "="*80)
        print("BOOTSTRAP ARROW STRENGTHS (ONE MODEL FIT PER RESAMPLE)")
        print("="*80)

        num_bootstrap_resamples = 200
        ci_confidence_level = 0.95
        alpha = 1.0 - ci_confidence_level
        lower_q = 100 * (alpha / 2.0)
        upper_q = 100 * (1.0 - alpha / 2.0)

        def fit_model_on_bootstrap_sample(boot_data: pd.DataFrame):
            boot_model = gcm.ProbabilisticCausalModel(causal_graph_all)
            gcm.auto.assign_causal_mechanisms(boot_model, boot_data)
            gcm.fit(boot_model, boot_data)
            return boot_model

        all_rows = []
        bootstrap_ve_sums = [[] for _ in range(num_bootstrap_resamples)]

        for target in target_nodes:
            target_parents = list(causal_graph_all.predecessors(target))
            if not target_parents:
                continue

            print(f"\nTarget: {target} | Parents: {target_parents}")

            ve_store = defaultdict(list)

            try:
                for b in range(num_bootstrap_resamples):
                    print(f"  Bootstrap resample {b+1}/{num_bootstrap_resamples}")
                    boot_idx = rng.integers(0, len(gcm_data), size=len(gcm_data))
                    boot_data = gcm_data.iloc[boot_idx].reset_index(drop=True)

                    boot_model = fit_model_on_bootstrap_sample(boot_data)

                    ve_strength_dict = gcm.arrow_strength(
                        boot_model,
                        target,
                        n_jobs=1
                    )

                    for parent in target_parents:
                        arrow_key = (parent, target)
                        ve_value = ve_strength_dict.get(arrow_key, np.nan)
                        ve_store[parent].append(float(ve_value))

                        if not np.isnan(ve_value):
                            bootstrap_ve_sums[b].append(float(ve_value))

                for parent in target_parents:
                    ve_vals = np.array(ve_store[parent], dtype=float)
                    ve_vals = ve_vals[~np.isnan(ve_vals)]

                    def calc_pvalue(bootstrap_vals):
                        if len(bootstrap_vals) == 0:
                            return np.nan
                        count_below_zero = np.sum(bootstrap_vals < 0)
                        count_above_zero = np.sum(bootstrap_vals > 0)
                        if count_below_zero == 0 or count_above_zero == 0:
                            return 1.0 / len(bootstrap_vals)
                        return 2.0 * min(count_below_zero, count_above_zero) / len(bootstrap_vals)

                    ve_pvalue = calc_pvalue(ve_vals)
                    ve_significant = "Yes" if ve_pvalue < 0.05 else "No"

                    all_rows.append({
                        "Parent": parent,
                        "Target": target,
                        "Arrow": f"{parent} -> {target}",
                        "Variance_Explained": np.mean(ve_vals) if len(ve_vals) else np.nan,
                        "VE_CI_95_Lower": np.percentile(ve_vals, lower_q) if len(ve_vals) else np.nan,
                        "VE_CI_95_Upper": np.percentile(ve_vals, upper_q) if len(ve_vals) else np.nan,
                        "VE_P_Value": ve_pvalue,
                        "VE_Significant": ve_significant,
                        "Bootstrap_Resamples": num_bootstrap_resamples,
                    })

            except Exception as e:
                print(f"Error for target {target}: {str(e)}")

        strength_df = pd.DataFrame(all_rows)
        if not strength_df.empty:
            strength_df = strength_df.sort_values(["Target", "Variance_Explained"], ascending=[True, False])

        strength_df.to_csv(strength_output_path, index=False)
        print(f"\nSaved arrow strengths with bootstrap CIs: {strength_output_path}")

        print_and_plot_arrow_strength(strength_df)
