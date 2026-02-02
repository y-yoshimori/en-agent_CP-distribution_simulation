"""Feature engineering utilities extracted from the notebook.

Functions:
- create_features(df): returns df with new engineered columns
"""

import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features used later in modeling and testing.

    Implements:
    - 転職経験
    - ランクギャップ
    - 登録経路（大）
    - 年収ギャップ＋/ー
    - エントリー数, エントリー業種一致率, エントリー職種一致率
    - 平均年収ギャップ＋/ー, メイン紹介経路
    - drop some interim columns if present
    """
    df = df.copy()

    # 転職経験
    if '転職回数' in df.columns:
        df['転職経験'] = df['転職回数'].apply(lambda x: '転職なし' if x == 0 else ('転職1回以上' if x >= 1 else x))

    # ランクギャップ
    rank_mapping = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
    if '人材ランク_x' in df.columns and '案件ランク' in df.columns:
        df['人材ランク_numeric'] = df['人材ランク_x'].map(rank_mapping)
        df['案件ランク_numeric'] = df['案件ランク'].map(rank_mapping)
        df['ランクギャップ'] = df['案件ランク_numeric'] - df['人材ランク_numeric']
        df = df.drop(columns=[c for c in ['人材ランク_numeric', '案件ランク_numeric'] if c in df.columns])

    # 登録経路（大）
    def categorize_registration_route(route):
        if pd.isna(route):
            return '他'
        route_str = str(route)
        if '（スカウト）' in route_str:
            return 'スカウト'
        elif '（新規会員）' in route_str:
            return '新規'
        elif '（案件応募）' in route_str:
            return '案件'
        else:
            return '他'

    if '登録経路' in df.columns:
        df['登録経路（大）'] = df['登録経路'].apply(categorize_registration_route)

    # 年収ギャップ
    if '求人年収上限（単位：万円）' in df.columns or '求人年収下限（単位：万円）' in df.columns:
        if '求人年収上限（単位：万円）' in df.columns and '現在年収' in df.columns:
            df['年収ギャップ＋'] = (df['求人年収上限（単位：万円）'] - df['現在年収']).fillna(0).astype(int)
        if '求人年収下限（単位：万円）' in df.columns and '現在年収' in df.columns:
            df['年収ギャップー'] = (df['現在年収'] - df['求人年収下限（単位：万円）']).fillna(0).astype(int)

    # エントリー数・業種一致率・職種一致率
    def check_industry_match(job_industry, experienced_industries_str):
        if pd.isna(job_industry) or pd.isna(experienced_industries_str):
            return 0
        experienced_industries = [s.strip() for s in str(experienced_industries_str).split(',')]
        return 1 if job_industry in experienced_industries else 0

    if '業種' in df.columns and '経験業種' in df.columns:
        df['業種一致'] = df.apply(lambda row: check_industry_match(row['業種'], row['経験業種']), axis=1).astype(int)

    if '職種' in df.columns and 'コア経験職種_x' in df.columns:
        df['職種一致'] = df.apply(lambda row: check_industry_match(row['職種'], row['コア経験職種_x']), axis=1).astype(int)

    if '求職者ID' in df.columns:
        agg = df.groupby('求職者ID').agg(
            エントリー数=('求職者ID', 'size'),
            業種一致の合計=('業種一致', 'sum') if '業種一致' in df.columns else pd.NamedAgg(column='求職者ID', aggfunc='size')
        ).reset_index()

        agg['エントリー業種一致率'] = (agg['業種一致の合計'] / agg['エントリー数']).fillna(0)
        df = pd.merge(df, agg[['求職者ID', 'エントリー数', 'エントリー業種一致率']], on='求職者ID', how='left')

        # 職種一致率
        if '職種一致' in df.columns:
            agg_job = df.groupby('求職者ID').agg(職種一致の合計=('職種一致', 'sum')).reset_index()
            agg_job['エントリー職種一致率'] = (agg_job['職種一致の合計'] / agg['エントリー数']).fillna(0)
            df = pd.merge(df, agg_job[['求職者ID', 'エントリー職種一致率']], on='求職者ID', how='left')

    # 平均年収ギャップ per 求職者ID
    if '年収ギャップ＋' in df.columns:
        avg_gap_plus = df.groupby('求職者ID')['年収ギャップ＋'].mean().reset_index().rename(columns={'年収ギャップ＋': '平均年収ギャップ＋'})
        df = pd.merge(df, avg_gap_plus, on='求職者ID', how='left')
    if '年収ギャップー' in df.columns:
        avg_gap_minus = df.groupby('求職者ID')['年収ギャップー'].mean().reset_index().rename(columns={'年収ギャップー': '平均年収ギャップー'})
        df = pd.merge(df, avg_gap_minus, on='求職者ID', how='left')

    # メイン紹介経路 (mode)
    if '紹介経路' in df.columns:
        mode_route = df.groupby('求職者ID')['紹介経路'].apply(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index().rename(columns={'紹介経路': 'メイン紹介経路'})
        df = pd.merge(df, mode_route, on='求職者ID', how='left')

    # Drop intermediate columns if present to mimic original notebook
    drop_cols = ['案件ランク', '求人ID', '企業', '年収ギャップ＋', '年収ギャップー', '業種一致', '職種一致']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df
