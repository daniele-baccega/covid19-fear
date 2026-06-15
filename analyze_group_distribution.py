import pandas as pd
import os

# Define the background columns
background_col = ["D1", "D2", "D8", "D9", "D12", "region_code", "infection_trend", "wave"]

# Load the final dataset
results_dir = 'results'
final_dataset_path = os.path.join(results_dir, 'causal_dataset_final.csv')

if not os.path.exists(final_dataset_path):
    print(f"Error: {final_dataset_path} not found. Please run extract_data.py first.")
else:
    # Load data
    data = pd.read_csv(final_dataset_path)
    
    print("\n" + "="*80)
    print("ROW COUNTS BY BACKGROUND GROUPS")
    print("="*80)
    
    # Group by all background columns and count rows
    group_counts = data.groupby(background_col, observed=True).size().reset_index(name='row_count')
    
    # Add average G1 (fear) for each group
    group_g1 = data.groupby(background_col, observed=True)['G1'].mean().reset_index(name='avg_G1')
    
    # Merge the two dataframes
    group_counts = group_counts.merge(group_g1, on=background_col)
    group_counts = group_counts.sort_values('row_count', ascending=False)
    
    print(f"\nTotal unique group combinations: {len(group_counts)}")
    print(f"Total rows in dataset: {len(data)}")
    print(f"\nTop 20 groups by row count:")
    print(group_counts[['row_count', 'avg_G1'] + background_col].head(20).to_string(index=False))
    
    # Summary statistics
    print(f"\n" + "-"*80)
    print("Summary Statistics:")
    print("-"*80)
    print(f"Mean rows per group: {group_counts['row_count'].mean():.2f}")
    print(f"Median rows per group: {group_counts['row_count'].median():.2f}")
    print(f"Min rows per group: {group_counts['row_count'].min()}")
    print(f"Max rows per group: {group_counts['row_count'].max()}")
    print(f"Std dev: {group_counts['row_count'].std():.2f}")
    
    # Distribution by individual background variables
    print(f"\n" + "-"*80)
    print("Distribution by Individual Variables:")
    print("-"*80)
    
    for col in background_col:
        print(f"\n{col}:")
        col_dist = data.groupby(col).agg({
            'G1': ['count', 'mean', 'std']
        }).reset_index()
        col_dist.columns = ['Value', 'Count', 'Avg_G1', 'Std_G1']
        col_dist = col_dist.sort_values('Count', ascending=False)
        print(col_dist.to_string(index=False))
        print(f"  Total unique values: {col_dist.shape[0]}")
    
    # Identify all possible combinations and missing ones
    print(f"\n" + "-"*80)
    print("Missing Combinations Analysis:")
    print("-"*80)
    
    # Get all unique values for each column
    unique_vals = {col: sorted(data[col].unique()) for col in background_col}
    
    # Calculate total possible combinations
    total_possible = 1
    for col in background_col:
        total_possible *= len(unique_vals[col])
    
    print(f"Total possible combinations: {total_possible}")
    print(f"Actual combinations with data: {len(group_counts)}")
    print(f"Missing combinations: {total_possible - len(group_counts)}")
    print(f"Coverage: {(len(group_counts) / total_possible * 100):.2f}%")
    
    # Show coverage by each variable
    print(f"\nUnique values per variable:")
    for col in background_col:
        n_unique = len(unique_vals[col])
        print(f"  {col}: {n_unique} values - {unique_vals[col]}")
    
    # Generate all possible combinations and find missing ones
    from itertools import product
    
    all_combinations = list(product(*[unique_vals[col] for col in background_col]))
    existing_combinations = set(map(tuple, group_counts[background_col].values))
    missing_combinations = [combo for combo in all_combinations if combo not in existing_combinations]
    
    print(f"\n" + "-"*80)
    print(f"Missing Group Combinations ({len(missing_combinations)} total):")
    print("-"*80)
    
    if missing_combinations:
        # Create a dataframe for better display
        missing_df = pd.DataFrame(missing_combinations, columns=background_col)
        print(missing_df.to_string(index=False))
    else:
        print("No missing combinations - all possible groups have data!")
    
    # Categorize groups by size
    print(f"\n" + "-"*80)
    print("Group Size Categories:")
    print("-"*80)
    
    size_categories = {
        'Large (>1000)': len(group_counts[group_counts['row_count'] > 1000]),
        'Medium (100-1000)': len(group_counts[(group_counts['row_count'] >= 100) & (group_counts['row_count'] <= 1000)]),
        'Small (10-99)': len(group_counts[(group_counts['row_count'] >= 10) & (group_counts['row_count'] < 100)]),
        'Very Small (1-9)': len(group_counts[group_counts['row_count'] < 10]),
    }
    
    for category, count in size_categories.items():
        percentage = (count / len(group_counts) * 100)
        print(f"  {category}: {count} groups ({percentage:.2f}%)")
    
    # Save detailed results
    output_path = os.path.join(results_dir, 'group_distribution_analysis.csv')
    group_counts.to_csv(output_path, index=False)
    print(f"\n" + "="*80)
    print(f"Detailed group distribution saved to: {output_path}")
    print("="*80)
    
    # Show groups sorted by average G1 (fear)
    print(f"\n" + "="*80)
    print("Groups Sorted by Average G1 (Fear Level):")
    print("="*80)
    
    group_counts_by_g1 = group_counts.sort_values('avg_G1', ascending=False)
    print(f"\nTop 15 groups with HIGHEST average fear (G1):")
    print(group_counts_by_g1[['avg_G1', 'row_count'] + background_col].head(15).to_string(index=False))
    
    print(f"\nTop 15 groups with LOWEST average fear (G1):")
    print(group_counts_by_g1[['avg_G1', 'row_count'] + background_col].tail(15).to_string(index=False))
    
    # Summary statistics for G1
    print(f"\n" + "-"*80)
    print("G1 Summary Statistics:")
    print("-"*80)
    print(f"Overall mean G1: {data['G1'].mean():.4f}")
    print(f"Overall variance G1: {data['G1'].var():.4f}")
    print(f"Overall std G1: {data['G1'].std():.4f}")
    print(f"Mean G1 across groups: {group_counts['avg_G1'].mean():.4f}")
    print(f"Std of group means: {group_counts['avg_G1'].std():.4f}")
    print(f"Min group avg G1: {group_counts['avg_G1'].min():.4f}")
    print(f"Max group avg G1: {group_counts['avg_G1'].max():.4f}")
    
    # Create summary statistics by each background variable
    print(f"\n" + "="*80)
    print("Average row count by each background variable:")
    print("="*80)
    
    for col in background_col:
        print(f"\n{col}:")
        var_summary = data.groupby(col).size().reset_index(name='total_rows')
        var_summary['avg_per_group'] = var_summary.groupby(col)['total_rows'].transform(lambda x: data[data[col].isin(x.index)].groupby(col).size().mean())
        for idx, row in var_summary.iterrows():
            print(f"  Value {row[col]}: {row['total_rows']} total rows")
