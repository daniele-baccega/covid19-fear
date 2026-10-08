import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pickle
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from dowhy import gcm, CausalModel

from utils import *

rng = np.random.default_rng(42)

def plot_arrow_strength_grid(
    df,
    panel_col="Target",
    label_col="Parent",
    value_col="Variance_Explained",
    ci_lower_col="VE_CI_95_Lower",
    ci_upper_col="VE_CI_95_Upper",
    sig_col="VE_Significant",
    nrows=3,
    ncols=3,
    figsize=(20, 14),
    sort_desc=True,
    top_n=None,
    percent_axis_right=True,
    rotate_xticks=35,
    color_sig="#76B7EB",
    color_nonsig="#C9D3DD",
    edgecolor="#222222",
    alpha=0.9,
    sharey=False,
    title=None,
    suptitle_y=0.98
):
    """
    Create a 3x3 grid of bar plots with asymmetric 95% CI error bars.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with one row per arrow.
    panel_col : str
        Column defining each subplot, typically 'Target'.
    label_col : str
        X-axis labels inside each subplot, typically 'Parent'.
    value_col : str
        Bar height column, typically 'Variance_Explained'.
    ci_lower_col : str
        Lower bound of 95% CI.
    ci_upper_col : str
        Upper bound of 95% CI.
    sig_col : str or None
        Optional column used to color significant vs non-significant bars.
        Expected values like 'Yes'/'No' or boolean.
    nrows, ncols : int
        Grid shape. Default is 3x3.
    figsize : tuple
        Figure size.
    sort_desc : bool
        Whether to sort bars within each panel by descending value.
    top_n : int or None
        If set, keep only top_n bars per panel.
    percent_axis_right : bool
        Add a secondary right y-axis with percentages.
    rotate_xticks : int
        Rotation angle for x tick labels.
    color_sig, color_nonsig : str
        Colors for significant/non-significant bars.
    edgecolor : str
        Bar edge color.
    alpha : float
        Bar transparency.
    sharey : bool
        Whether subplots share y axis.
    title : str or None
        Overall figure title.
    suptitle_y : float
        Vertical position of figure title.

    Returns
    -------
    fig, axes
    """
    df = df.copy()

    needed = [panel_col, label_col, value_col, ci_lower_col, ci_upper_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[panel_col, label_col, value_col, ci_lower_col, ci_upper_col])

    panels = list(pd.unique(df[panel_col]))
    max_panels = nrows * ncols
    if len(panels) > max_panels:
        raise ValueError(
            f"Found {len(panels)} panels but grid allows only {max_panels}. "
            f"Increase nrows/ncols or filter the dataframe."
        )

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()

    global_ymax = 0
    prepared = {}

    for panel in panels:
        sub = df[df[panel_col] == panel].copy()

        if sort_desc:
            sub = sub.sort_values(value_col, ascending=False)

        if top_n is not None:
            sub = sub.head(top_n)

        y = sub[value_col].to_numpy()
        yerr_lower = y - sub[ci_lower_col].to_numpy()
        yerr_upper = sub[ci_upper_col].to_numpy() - y

        yerr_lower = np.clip(yerr_lower, 0, None)
        yerr_upper = np.clip(yerr_upper, 0, None)

        ymax_panel = np.nanmax(sub[ci_upper_col].to_numpy()) if len(sub) else 0
        global_ymax = max(global_ymax, ymax_panel)

        prepared[panel] = (sub, y, yerr_lower, yerr_upper)

    if global_ymax <= 0:
        global_ymax = 1.0
    global_ymax *= 1.12

    for i, panel in enumerate(panels):
        ax = axes[i]
        sub, y, yerr_lower, yerr_upper = prepared[panel]

        if sig_col is not None and sig_col in sub.columns:
            sig_vals = sub[sig_col].astype(str).str.lower()
            colors = [
                color_sig if v in {"yes", "true", "1"} else color_nonsig
                for v in sig_vals
            ]
        else:
            colors = [color_sig] * len(sub)

        x = np.arange(len(sub))

        ax.bar(
            x,
            y,
            color=colors,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=1.2,
            yerr=np.vstack([yerr_lower, yerr_upper]),
            ecolor="#3A3A3A",
            capsize=5
        )

        ax.set_title(str(panel), fontsize=13, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(sub[label_col].astype(str), rotation=rotate_xticks, ha="right")
        ax.set_ylim(0, global_ymax)
        ax.grid(axis="y", alpha=0.25, linestyle="-")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i % ncols == 0:
            ax.set_ylabel("Variance explained")

        if percent_axis_right:
            axr = ax.twinx()
            axr.set_ylim(ax.get_ylim())
            axr.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
            axr.set_ylabel("Variance explained (%)")
            axr.grid(False)

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=18, y=suptitle_y)

    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    return fig, axes

# =====================================================
# DATA EXTRACTION AND LOADING
# =====================================================

data_causal_model = load_or_extract_datasets(
    causal_csv_path='data/causal_dataset.csv',
    extract_func=extract_and_prepare_datasets
)
data_causal_model = add_infection_from_sird(data_causal_model, 'data/SIRD.RData', 'EndDatetime')

results_dir = 'results_G1_I5'
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

    data_causal_model = add_infection_from_sird(data_causal_model, 'data/SIRD.RData', 'EndDatetime')

    # Fallback to legacy path, then default to zeros to avoid downstream KeyError.
    if 'Infection' not in data_causal_model.columns:
        data_causal_model = add_infection_from_sird(data_causal_model, 'SIRD.RData', 'EndDatetime')
    if 'Infection' not in data_causal_model.columns:
        print("Warning: Infection column unavailable after SIRD merge attempts. Filling with 0.0.")
        data_causal_model['Infection'] = 0.0

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
    ax.set_xlabel('', fontsize=12)
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

    restrictions_all = pd.read_csv("data/OxCGRT_compact_subnational_v1.csv")
    restrictions_all["Date"] = pd.to_datetime(restrictions_all["Date"], format="%Y%m%d")

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

    zipcode_mapping_path = "utils/uszips.csv"
    region_mapping_path = "utils/county_fips_master.csv"
    state_mapping = load_zipcode_mapping(zipcode_mapping_path)
    region_mapping = load_region_mapping(region_mapping_path)
    restrictions_all = map_states_to_regions(restrictions_all, region_mapping)
    state_numeric_mapping, region_numeric_mapping = create_numeric_encodings(state_mapping, region_mapping)
    restrictions_all = apply_numeric_encodings(restrictions_all, state_numeric_mapping, region_numeric_mapping)
    restrictions_all = restrictions_all.groupby(['Date', 'region_code'], as_index=False)["StringencyIndex_Average"].mean()
    restrictions_all.to_csv(os.path.join(results_dir, 'restrictions_data.csv'), index=False)

    plt.figure(figsize=(14, 6))
    for region_name, group_data in restrictions_all.groupby('region_code'):
        plt.plot(group_data['Date'], group_data['StringencyIndex_Average'], label=f"Region: {region_name}")
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    restrictions_dist_path = os.path.join(results_dir, 'distribution_restrictions.png')
    plt.savefig(restrictions_dist_path, dpi=300, bbox_inches='tight')
    print(f"Saved restrictions distribution plot: {restrictions_dist_path}")
    plt.close()

    data_causal_model['EndDatetime'] = pd.to_datetime(data_causal_model['EndDatetime'], errors='coerce')
    restrictions_all['Date'] = pd.to_datetime(restrictions_all['Date'], errors='coerce')

    data_causal_model = data_causal_model.merge(
        restrictions_all[['Date', 'StringencyIndex_Average']],
        left_on='EndDatetime',
        right_on='Date',
        how='left'
    ).drop(columns=['Date'])

    data_causal_model['StringencyIndex_Average'] = pd.cut(
        data_causal_model['StringencyIndex_Average'],
        bins=np.arange(0, 110, 10),
        labels=np.arange(1, 11),
        right=False
    ).astype(float)

    data_causal_model.to_csv(final_dataset_path, index=False)
    data_causal_model.head(n=1000).to_csv(
        os.path.join(results_dir, 'causal_dataset_final_sample.csv'),
        index=False
    )

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
edges += [("G1", source_col) for source_col in source_cols]
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
print("Estimating causal effects of Fear (G1) on Information Sources (I5)")
print("="*80)

ate_data = data_causal_model.copy()
ate_results = []

treatment_var = "G1"
all_confounders = ["D1", "D2", "D8", "D9", "D12", "region_code", "wave", "infection_trend", "StringencyIndex_Average"]

# Determine if treatment is binary or multi-valued
n_treatment_values = ate_data[treatment_var].nunique()
is_binary = n_treatment_values == 2
method_type = "Binary (propensity score)" if is_binary else "Multi-valued (linear regression)"

print("\n" + "-"*80)
print(f"ATE for {treatment_var} → Information Sources (I5)")
print(f"Treatment type: {method_type}")
print(f"Number of treatment values: {n_treatment_values}")
print("-"*80)

# Estimate G1's effect on each source variable
for outcome_var in source_cols:
    if outcome_var not in ate_data.columns:
        continue

    try:
        model = CausalModel(
            data=ate_data,
            treatment=treatment_var,
            outcome=outcome_var,
            common_causes=all_confounders,
            graph=causal_graph_all
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # Choose method based on treatment type
        if is_binary:
            method = "backdoor.propensity_score_stratification"
        else:
            method = "backdoor.linear_regression"
        
        ate_estimate = model.estimate_effect(identified_estimand, method_name=method)

        print(f"{outcome_var:15s}: ATE = {ate_estimate.value:+.6f}")

        ate_results.append({
            'Treatment': treatment_var,
            'Outcome': outcome_var,
            'ATE': ate_estimate.value,
            'Method': method.split('.')[-1],
            'N_Confounders': len(all_confounders)
        })

    except Exception as e:
        print(f"{outcome_var:15s}: Error - {str(e)[:80]}")

if ate_results:
    ate_df = pd.DataFrame(ate_results).sort_values('ATE', key=abs, ascending=False)
    ate_df.to_csv(os.path.join(results_dir, 'ate_estimation_results.csv'), index=False)

    print(f"\n{'='*80}\nATE Results Saved\n{'='*80}")
    print(ate_df[['Treatment', 'Outcome', 'ATE', 'Method']].to_string(index=False))

    print(f"\n{'='*80}\nSummary Statistics\n{'='*80}")
    print(f"Mean ATE (Fear → Sources): {ate_df['ATE'].mean():.6f}")
    print(f"Std ATE: {ate_df['ATE'].std():.6f}")
    print(f"Min ATE: {ate_df['ATE'].min():.6f}")
    print(f"Max ATE: {ate_df['ATE'].max():.6f}")
    print(f"\nPositive effects (n={len(ate_df[ate_df['ATE'] > 0])}): Sources where higher fear increases usage")
    print(f"Negative effects (n={len(ate_df[ate_df['ATE'] < 0])}): Sources where higher fear decreases usage")
    print(f"\n{'='*80}")

    # =====================================================
    # K-MEANS CLUSTERING OF ATE VALUES
    # =====================================================

    print(f"\n{'='*80}\nK-MEANS CLUSTERING OF ATE VALUES\n{'='*80}")

    # Prepare data for clustering
    ate_values = ate_df[['ATE']].values
    scaler = StandardScaler()
    ate_scaled = scaler.fit_transform(ate_values)

    # Test different numbers of clusters
    k_range = range(2, 5)
    silhouette_scores = []
    kmeans_models = {}

    print(f"\nEvaluating k-means for k = 2 to 5...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(ate_scaled)
        sil_score = silhouette_score(ate_scaled, cluster_labels)
        silhouette_scores.append(sil_score)
        kmeans_models[k] = kmeans
        print(f"  k={k}: Silhouette Score = {sil_score:.4f}")

    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    optimal_score = max(silhouette_scores)
    print(f"\nOptimal number of clusters: k={optimal_k} (Silhouette Score: {optimal_score:.4f})")

    # Save cluster info for the top-2 k values by silhouette score.
    k_to_score = dict(zip(k_range, silhouette_scores))
    top_2_k = sorted(k_to_score, key=k_to_score.get, reverse=True)[:2]
    print(f"Top-2 k by silhouette score: {top_2_k}")

    top2_summary_rows = []
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

    top2_summary_path = os.path.join(results_dir, 'ate_top2_k_silhouette_scores.csv')
    pd.DataFrame(top2_summary_rows).to_csv(top2_summary_path, index=False)
    print(f"Saved top-2 k silhouette summary: {top2_summary_path}")

    # Fit final model with optimal k
    final_kmeans = kmeans_models[optimal_k]
    ate_df['Cluster'] = final_kmeans.predict(ate_scaled)

    # Add cluster sizes and centroid distances
    cluster_sizes = ate_df['Cluster'].value_counts().sort_index()
    print(f"\nCluster Distribution:")
    for cluster_id in sorted(ate_df['Cluster'].unique()):
        cluster_data = ate_df[ate_df['Cluster'] == cluster_id]
        print(f"  Cluster {cluster_id}: n={len(cluster_data)}, Mean ATE={cluster_data['ATE'].mean():.6f}, Std={cluster_data['ATE'].std():.6f}")

    # Save clustered results
    ate_clustered_path = os.path.join(results_dir, 'ate_estimation_results_clustered.csv')
    ate_df.to_csv(ate_clustered_path, index=False)
    print(f"\nSaved clustered ATE results: {ate_clustered_path}")

    # Create silhouette score plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
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

    # Create scatter plot of ATE values colored by cluster
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
    for cluster_id in sorted(ate_df['Cluster'].unique()):
        cluster_data = ate_df[ate_df['Cluster'] == cluster_id]
        ax.scatter(range(len(cluster_data)), cluster_data['ATE'].values, 
                  c=[colors[cluster_id]], label=f'Cluster {cluster_id}', 
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Treatment Index', fontsize=12)
    ax.set_ylabel('ATE Value', fontsize=12)
    ax.set_title(f'ATE Values by K-Means Cluster (k={optimal_k})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    cluster_scatter_path = os.path.join(results_dir, 'ate_cluster_scatter.png')
    plt.savefig(cluster_scatter_path, dpi=300, bbox_inches='tight')
    print(f"Saved cluster scatter plot: {cluster_scatter_path}")
    plt.close()

    print(f"\n{'='*80}")

# =====================================================
# ESTIMATE OF ARROW STRENGTH
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

    target_nodes = source_cols  # Now source variables are the outcomes

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
    
    target_summary_rows = []

    target_nodes = ["I5_3", "I5_4", "I5_5", "I5_6", "I5_7", "I5_8", "I5_9"]

    for target in target_nodes:
        target_parents = list(causal_graph_all.predecessors(target))
        if not target_parents:
            continue

        print(f"\nTarget: {target} | Parents: {target_parents}")

        md_store = defaultdict(list)
        ve_store = defaultdict(list)
        target_rows = []  # Rows specific to this target
        target_ve_sums = [[] for _ in range(num_bootstrap_resamples)]  # VE sums for this target

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
                    
                    # Accumulate to target-specific sums (skip NaN values)
                    if not np.isnan(ve_value):
                        target_ve_sums[b].append(float(ve_value))

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

                target_rows.append({
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
        
        # Save each target's arrow strengths to separate file
        if target_rows:
            target_strength_df = pd.DataFrame(target_rows)
            target_strength_df = target_strength_df.sort_values("Variance_Explained", ascending=False)
            target_output_path = os.path.join(
                results_dir,
                f"arrow_strength_{target}.csv"
            )
            target_strength_df.to_csv(target_output_path, index=False)
            print(f"Saved {target} arrow strengths: {target_output_path}")
            
            # Compute total variance explained for this target
            target_ve_sums_array = np.array([np.sum(sums) for sums in target_ve_sums])
            target_ve_sums_array = target_ve_sums_array[~np.isnan(target_ve_sums_array) & ~np.isinf(target_ve_sums_array)]

            if len(target_ve_sums_array) > 0:
                target_ve_mean = np.mean(target_ve_sums_array)
                target_ve_ci_lower = np.percentile(target_ve_sums_array, lower_q)
                target_ve_ci_upper = np.percentile(target_ve_sums_array, upper_q)
                target_ve_std = np.std(target_ve_sums_array)
                
                def calc_pvalue_target(bootstrap_vals):
                    if len(bootstrap_vals) == 0:
                        return np.nan
                    count_below_zero = np.sum(bootstrap_vals < 0)
                    count_above_zero = np.sum(bootstrap_vals > 0)
                    if count_below_zero == 0 or count_above_zero == 0:
                        return 1.0 / len(bootstrap_vals)
                    return 2.0 * min(count_below_zero, count_above_zero) / len(bootstrap_vals)
                
                target_ve_pvalue = calc_pvalue_target(target_ve_sums_array)
                target_ve_significant = "Yes" if target_ve_pvalue < 0.05 else "No"
                
                target_summary_rows.append({
                    "Target": target,
                    "Total_VE_Mean": target_ve_mean,
                    "Total_VE_Std": target_ve_std,
                    "Total_VE_CI_95_Lower": target_ve_ci_lower,
                    "Total_VE_CI_95_Upper": target_ve_ci_upper,
                    "Total_VE_P_Value": target_ve_pvalue,
                    "Total_VE_Significant": target_ve_significant,
                    "N_Parents": len(target_parents),
                    "Parents": ", ".join(target_parents),
                    "Bootstrap_Samples": len(target_ve_sums_array),
                    "Total_Bootstrap_Resamples": num_bootstrap_resamples
                })
                
                print(f"  Total VE for {target}: {target_ve_mean:.6f} [{target_ve_ci_lower:.6f}, {target_ve_ci_upper:.6f}]")

    # Save summary of total VE per target
    if target_summary_rows:
        target_summary_df = pd.DataFrame(target_summary_rows).sort_values("Total_VE_Mean", ascending=False)
        target_summary_path = os.path.join(
            results_dir,
            "arrow_strength_summary_per_target.csv"
        )
        target_summary_df.to_csv(target_summary_path, index=False)
        print(f"\nSaved summary of total VE per target: {target_summary_path}")

    print("\n" + "="*80)
    print("ARROW STRENGTH ANALYSIS COMPLETE")
    print("="*80)
    print(f"Generated separate files for each target node.")
