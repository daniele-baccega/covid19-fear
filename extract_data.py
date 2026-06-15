import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import seaborn as sns
from collections import defaultdict

from dowhy import gcm, CausalModel

from utils import *

rng = np.random.default_rng(42)

# =====================================================
# DATA EXTRACTION AND LOADING
# =====================================================

data_causal_model = load_or_extract_datasets(
    causal_csv_path='causal_dataset.csv',
    extract_func=extract_and_prepare_datasets
)
data_causal_model = add_infection_from_sird(data_causal_model, 'SIRD.RData', 'EndDatetime')

results_dir = 'results_I5_G1'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

final_dataset_path = os.path.join(results_dir, 'causal_dataset_final.csv')
if os.path.exists(final_dataset_path):
    print(f"Skipping final post-processing: existing file found at {final_dataset_path}")
    data_causal_model = pd.read_csv(final_dataset_path)
else:
    data_causal_model = data_causal_model.dropna(subset=['G1'])

    i5_cols = [f'I5_{i}' for i in range(1, 10)]
    data_dummies = data_causal_model['I5'].fillna('').str.get_dummies(sep=',').add_prefix('I5_')

    for col in i5_cols:
        data_causal_model[col] = data_dummies[col].astype(float) if col in data_dummies.columns else 0.0

    data_causal_model = data_causal_model.drop(columns="I5")

    data_causal_model = add_infection_from_sird(data_causal_model, 'SIRD.RData', 'EndDatetime')

    for col in data_causal_model.columns:
        if col not in ['EndDatetime', 'Infection']:
            data_causal_model[col] = pd.to_numeric(data_causal_model[col], errors='coerce')
            # data_causal_model[col] = data_causal_model[col].fillna(0).astype(int)

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

    # infection_by_date = data_causal_model.groupby('EndDatetime')['Infection'].mean().sort_index()

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
    ax.set_title('Infections Over Time with Wave Cutoff and Trend', fontsize=13, fontweight='bold')
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

    data_causal_model.to_csv(final_dataset_path, index=False)
    data_causal_model.head(n=1000).to_csv(
        os.path.join(results_dir, 'causal_dataset_final_sample.csv'),
        index=False
    )

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
    'I5_1': 'Doctors',
    'I5_2': 'Scientists',
    'I5_3': 'CDC',
    'I5_4': 'Government',
    'I5_5': 'Politicians',
    'I5_6': 'Journalists',
    'I5_7': 'Family and friends',
    'I5_8': 'Religious leaders',
    'I5_9': 'None of the above'
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
    i5_df.plot(kind='bar', stacked=True, ax=ax, color=['#B3E5B3', '#FFB3B3'], edgecolor='black', width=0.7)
    
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_xticklabels(i5_df.index, rotation=30, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend([label_mappings['I5'].get(v, str(v)) for v in i5_df.columns], title='Usage', loc='upper right', fontsize=12, title_fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    i5_dist_path = os.path.join(results_dir, 'distribution_i5_sources.png')
    plt.savefig(i5_dist_path, dpi=300, bbox_inches='tight')
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
    i6_df.plot(kind='bar', stacked=True, ax=ax, color=['#B3E5B3', '#FFFFCC', '#FFB3B3'], edgecolor='black', width=0.7)
    
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_xticklabels(i6_df.index, rotation=30, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(['Trust', 'Somewhat', 'Not trust'], title='Trust Level', loc='upper right', fontsize=12, title_fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    i6_dist_path = os.path.join(results_dir, 'distribution_i6_trust.png')
    plt.savefig(i6_dist_path, dpi=300, bbox_inches='tight')
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
    n_cols = 3
    n_rows = (n_demos + n_cols - 1) // n_cols  # Calculate rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6))
    axes = axes.flatten()  # Flatten to 1D for easier iteration
    
    for idx, col in enumerate(demo_cols_exist):
        value_counts = data_causal_model[col].value_counts().sort_index()
        percentages = (value_counts / value_counts.sum()) * 100
        axes[idx].set_title(var_names.get(col, col), fontsize=12, fontweight='bold')
        axes[idx].bar(percentages.index, percentages.values, color='#D9F0D9', edgecolor='black')
        # Only show y-axis label on first plot of each row
        if idx % n_cols == 0:
            axes[idx].set_ylabel('Percentage (%)', fontsize=13)
        axes[idx].set_xticks(sorted(percentages.index))
        if col == 'D8':
            # Split Education labels on 2 rows
            d8_labels = [label_mappings[col].get(v, str(v)) for v in sorted(percentages.index)]
            d8_labels_wrapped = [label.replace('/', '/\n').replace(' or ', '\nor ') for label in d8_labels]
            axes[idx].set_xticklabels(d8_labels_wrapped, fontsize=13)
        else:
            axes[idx].set_xticklabels([label_mappings[col].get(v, str(v)) for v in sorted(percentages.index)], fontsize=13)            
        axes[idx].tick_params(axis='y', labelsize=12)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Hide any unused subplots
    for idx in range(len(demo_cols_exist), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    demo_dist_path = os.path.join(results_dir, 'distribution_demographics.png')
    plt.savefig(demo_dist_path, dpi=300, bbox_inches='tight')
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
    axes[1].bar(v1_pct.index, v1_pct.values, color='#FFFFE0', edgecolor='black')
    axes[1].set_ylabel('Percentage (%)', fontsize=13)
    axes[1].set_xticks(sorted(v1_pct.index))
    axes[1].set_xticklabels([label_mappings['V1'].get(v, str(v)) for v in sorted(v1_pct.index)], fontsize=13)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')

# A4 (Continuous 0-100) - Histogram with %
if 'A4' in data_causal_model.columns:
    axes[2].set_title('Sick', fontsize=12, fontweight='bold')
    axes[2].hist(data_causal_model['A4'], bins=30, color='#B3D9FF', edgecolor='black', alpha=0.7, weights=np.ones(len(data_causal_model))/len(data_causal_model)*100)
    axes[2].set_ylabel('Percentage (%)', fontsize=13)
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

print("\n" + "="*80)
print("CAUSAL INFERENCE - FULL DATASET ANALYSIS")
print("Using built graph and all available rows")
print("="*80)

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

    # =====================================================
    # BOOTSTRAP ARROW STRENGTHS WITH BUILT-IN CONFIDENCE INTERVALS
    # =====================================================

    print("\n" + "="*80)
    print("BOOTSTRAP ARROW STRENGTHS (ONE MODEL FIT PER RESAMPLE)")
    print("="*80)

    num_bootstrap_resamples = 500
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
    
    # Track sum of VE across all parent-target pairs for each bootstrap resample
    bootstrap_ve_sums = [[] for _ in range(num_bootstrap_resamples)]

    for target in target_nodes:
        target_parents = list(causal_graph_all.predecessors(target))
        if not target_parents:
            continue

        print(f"\nTarget: {target} | Parents: {target_parents}")

        md_store = defaultdict(list)
        ve_store = defaultdict(list)

        try:
            for b in range(num_bootstrap_resamples):
                print(f"  Bootstrap resample {b+1}/{num_bootstrap_resamples}", end='\r')
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
                    
                    # Accumulate to bootstrap sums (skip NaN values)
                    if not np.isnan(ve_value):
                        bootstrap_ve_sums[b].append(float(ve_value))

            for parent in target_parents:
                ve_vals = np.array(ve_store[parent], dtype=float)
                ve_vals = ve_vals[~np.isnan(ve_vals)]

                # Calculate p-values from bootstrap distributions
                def calc_pvalue(bootstrap_vals):
                    """Two-tailed p-value: proportion of bootstrap samples on opposite side of zero"""
                    if len(bootstrap_vals) == 0:
                        return np.nan
                    count_below_zero = np.sum(bootstrap_vals < 0)
                    count_above_zero = np.sum(bootstrap_vals > 0)
                    if count_below_zero == 0 or count_above_zero == 0:
                        # All on one side: p-value very small
                        return 1.0 / len(bootstrap_vals)
                    return 2.0 * min(count_below_zero, count_above_zero) / len(bootstrap_vals)

                ve_pvalue = calc_pvalue(ve_vals)

                # Significance flags (α = 0.05)
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

    strength_output_path = os.path.join(
        results_dir,
        "causal_inference_full_dataset_arrow_strength.csv"
    )
    strength_df.to_csv(strength_output_path, index=False)
    print(f"\nSaved arrow strengths with bootstrap CIs: {strength_output_path}")

    # =====================================================
    # COMPUTE TOTAL VARIANCE EXPLAINED AND 95% CI
    # =====================================================
    
    print("\n" + "="*80)
    print("TOTAL VARIANCE EXPLAINED (Sum of Arrow Strengths)")
    print("="*80)
    
    # Compute sums for each bootstrap resample
    ve_sums = np.array([np.sum(sums) for sums in bootstrap_ve_sums])
    
    # Remove NaN or inf values
    ve_sums = ve_sums[~np.isnan(ve_sums) & ~np.isinf(ve_sums)]
    
    # Compute statistics
    ve_mean = np.mean(ve_sums)
    ve_ci_lower = np.percentile(ve_sums, lower_q)
    ve_ci_upper = np.percentile(ve_sums, upper_q)
    ve_std = np.std(ve_sums)
    
    # Compute p-value: proportion of bootstrap samples on opposite side of zero
    def calc_pvalue_ve(bootstrap_vals):
        """Two-tailed p-value for variance explained"""
        if len(bootstrap_vals) == 0:
            return np.nan
        count_below_zero = np.sum(bootstrap_vals < 0)
        count_above_zero = np.sum(bootstrap_vals > 0)
        if count_below_zero == 0 or count_above_zero == 0:
            return 1.0 / len(bootstrap_vals)
        return 2.0 * min(count_below_zero, count_above_zero) / len(bootstrap_vals)
    
    ve_pvalue = calc_pvalue_ve(ve_sums)
    ve_significant = "Yes" if ve_pvalue < 0.05 else "No"
    
    print(f"\nVariance Explained (Sum across all arrows):")
    print(f"  Mean:          {ve_mean:.6f}")
    print(f"  Std Dev:       {ve_std:.6f}")
    print(f"  95% CI:        [{ve_ci_lower:.6f}, {ve_ci_upper:.6f}]")
    print(f"  P-Value:       {ve_pvalue:.6f}")
    print(f"  Significant:   {ve_significant}")
    print(f"  Bootstrap Samples with valid sums: {len(ve_sums)}/{num_bootstrap_resamples}")
    
    # Save summary statistics
    summary_rows = [
        {
            "Metric": "Total Variance Explained",
            "Mean": ve_mean,
            "Std_Dev": ve_std,
            "CI_95_Lower": ve_ci_lower,
            "CI_95_Upper": ve_ci_upper,
            "P_Value": ve_pvalue,
            "Significant": ve_significant,
            "Bootstrap_Samples": len(ve_sums),
            "Total_Bootstrap_Resamples": num_bootstrap_resamples
        }
    ]
    
    summary_df = pd.DataFrame(summary_rows)
    summary_output_path = os.path.join(
        results_dir,
        "causal_inference_total_arrow_strength_summary.csv"
    )
    summary_df.to_csv(summary_output_path, index=False)
    print(f"\nSaved total arrow strength summary: {summary_output_path}")


# =====================================================
# ATE ESTIMATION
# =====================================================

print("\n" + "="*80)
print("AVERAGE TREATMENT EFFECT (ATE) ESTIMATION")
print("Estimating causal effects of I5/I6 sources on Fear (G1)")
print("="*80)

ate_data = data_causal_model.copy()
ate_results = []

print("\n" + "-"*80)
print("ATE for All Variables → Fear (G1)")
print("Binary treatments: propensity score | Multi-valued: linear regression")
print("-"*80)

binary_vars = source_cols + ["V1"]
multivalue_vars = ["A4"]
all_treatment_vars = binary_vars + multivalue_vars
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
    ate_df.to_csv(os.path.join(results_dir, 'ate_estimation_results.csv'), index=False)

    print(f"\n{'='*80}\nATE Results Saved\n{'='*80}")
    print(ate_df[['Treatment', 'ATE', 'Category', 'Method']].to_string(index=False))

    print(f"\n{'='*80}\nSummary by Category\n{'='*80}")
    for category in ['Demographic', 'Information Source (I5)', 'Trust (I6)', 'Other']:
        cat_data = ate_df[ate_df['Category'] == category]
        if len(cat_data) > 0:
            print(f"\n{category} (n={len(cat_data)}): Mean ATE = {cat_data['ATE'].mean():.6f}")
            top_3 = cat_data.sort_values('ATE', key=abs, ascending=False).head(3)
            for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                print(f"  {idx}. {row['Treatment']:15s}: {row['ATE']:+.6f} ({row['Method']})")
    print(f"\n{'='*80}")