"""Statistical tests (Mann-Whitney U) utilities.

Function:
- run_mannwhitney_tests(filtered_cp_dataframes_list, features_to_test)

Returns a DataFrame with statistical results similar to the original notebook.
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu


def run_mannwhitney_tests(filtered_cp_dataframes_list, features_to_test):
    results = []

    for df_cp in filtered_cp_dataframes_list:
        if df_cp.empty:
            continue
        cp_name = df_cp['担当CP'].iloc[0]
        overall_mean_entries = df_cp['エントリー数'].mean()

        for feature in features_to_test:
            if feature not in df_cp.columns:
                continue

            feature_stats = df_cp.groupby(feature)['エントリー数'].agg(mean_entries='mean', sample_count='count').reset_index()

            high_performing_categories = feature_stats[(feature_stats['mean_entries'] >= 1.1 * overall_mean_entries) & (feature_stats['sample_count'] >= 4)]

            for _, row in high_performing_categories.iterrows():
                category_value = row[feature]
                mean_entries_in_category = row['mean_entries']
                sample_count_in_category = row['sample_count']

                group_a_data = df_cp[df_cp[feature] == category_value]['エントリー数'].values
                group_b_data = df_cp[df_cp[feature] != category_value]['エントリー数'].values

                if len(group_a_data) > 1 and len(group_b_data) > 1:
                    statistic, p_value = mannwhitneyu(group_a_data, group_b_data, alternative='two-sided')
                    N = len(group_a_data) + len(group_b_data)
                    n1 = len(group_a_data)
                    n2 = len(group_b_data)

                    if n1 == 0 or n2 == 0:
                        effect_size_r = np.nan
                    else:
                        U = statistic
                        expected_U = (n1 * n2) / 2
                        std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                        if std_U != 0:
                            z_score = (U - expected_U) / std_U
                            effect_size_r = z_score / np.sqrt(N)
                        else:
                            effect_size_r = 0.0 if U == expected_U else np.nan

                    results.append({
                        'CP': cp_name,
                        'Feature': feature,
                        'Category': category_value,
                        'Mean_Entries_in_Category': mean_entries_in_category,
                        'p_value': p_value,
                        'effect_size': effect_size_r,
                        'Overall_Mean_Entries_for_CP': overall_mean_entries,
                        'Sample_Count_in_Category': sample_count_in_category
                    })

    return pd.DataFrame(results)
