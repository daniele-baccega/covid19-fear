import pandas as pd
import numpy as np

df = pd.read_csv('results/causal_dataset_final.csv')

print("="*80)
print("MISSING VALUES SUMMARY")
print("="*80)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Total missing values: {df.isna().sum().sum()}")
print(f"Total cells: {len(df) * len(df.columns)}")
print(f"Percentage missing: {(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%")

# Show columns with missing values
missing_by_col = df.isna().sum()
cols_with_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
if len(cols_with_missing) > 0:
    print("\nColumns with missing values:")
    for col, count in cols_with_missing.items():
        pct = (count / len(df) * 100)
        print(f"  {col:20s}: {count:7d} missing ({pct:6.2f}%)")
else:
    print("\nNo missing values found!")

# Define all variables
source_cols = [f'I5_{i}' for i in range(1, 10)]
trust_cols = [f'I6_{i}' for i in range(1, 9)]
demo_cols = ['D1', 'D2', 'D8', 'D9', 'D12', 'region_code']
temporal_cols = ['infection_trend', 'wave']
outcome_col = ['G1']
other_cols = ['V1', 'A4']

print("="*80)
print("UNIQUE VALUE COUNTS FOR ALL VARIABLES")
print("="*80)

categories = {
    'Information Sources (I5)': source_cols,
    'Trust Ratings (I6)': trust_cols,
    'Demographics': demo_cols,
    'Temporal': temporal_cols,
    'Other Variables': other_cols,
    'Outcome': outcome_col
}

for category, cols in categories.items():
    print(f"\n{category}")
    print("-" * 80)
    for col in cols:
        if col in df.columns:
            unique_count = df[col].nunique()
            missing = df[col].isna().sum()
            print(f"  {col:20s}: {unique_count:3d} unique values, {missing:7d} missing")
            if unique_count <= 100:
                value_counts = df[col].value_counts().sort_index()
                for val, count in value_counts.items():
                    print(f"    {val:10.1f}: {count:10d}")
                
            # Print avg and std
            print(f"    Average: {df[col].mean():.2f}, Std Dev: {df[col].std():.2f}")
        else:
            print(f"  {col:20s}: NOT FOUND")
