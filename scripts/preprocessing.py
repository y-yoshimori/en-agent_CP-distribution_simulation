"""Preprocessing utilities extracted from the notebook.

Functions:
- preprocess_merge(df_mendan, df_oubo, excluded_mendan_cols=None, columns_to_drop=None, rename_map=None)
- add_time_deltas(df_merge_diff, df_mendan, df_oubo)
- finalize_dataset(df_merge_diff, df_mendan, df_oubo, df_seiyaku)

These functions aim to reproduce the notebook behavior in a modular way.
"""

import pandas as pd
import numpy as np


def preprocess_merge(df_mendan: pd.DataFrame, df_oubo: pd.DataFrame,
                     excluded_mendan_cols=None, columns_to_drop=None, rename_map=None):
    """Merge df_oubo with a filtered df_mendan and apply initial column drops/renames.

    Returns: df_merge_diff (DataFrame)
    """
    if excluded_mendan_cols is None:
        excluded_mendan_cols = [
            '人材担当', '求職者生年月日', '面談日時', '面談月', '面談週', '年齢帯', '★登録経路',
            '面談カウント対象', '★経験職種', '面談フラグ', '面談所属判別用', '面談所属div',
            '面談所属T', '面談予実用登録経路', '26期首都圏', '首都圏配布', '暫定用組織',
            '★キックオフ用領域'
        ]

    # Keep 求職者ID always
    mendan_cols_to_keep = [col for col in df_mendan.columns if col not in excluded_mendan_cols or col == '求職者ID']
    df_mendan_filtered = df_mendan[mendan_cols_to_keep].copy()

    df_merged = pd.merge(df_oubo, df_mendan_filtered, on='求職者ID', how='left')

    if columns_to_drop is None:
        columns_to_drop = [
            '求職者登録経路', '求職者生年月日', '年齢帯', '生年月日', '求職者年収帯', '★登録経路',
            '登録経路.1', '6/16追加（土日祝日集計漏れ分）', '「1次面接設定中」のステータス登録日',
            '最終面接日', '本人意思確認待ちステータス日付', '入社実績入力待ちステータス日付',
            '求職者登録経路.1', 'エンエージェントジャッジ', '人材ランク_y', '面談週', 'EC書類提出日',
            'コア経験職種_y', '進捗ID', '担当ECの所属部署', '担当ECの所属チーム', '担当EC',
            '担当CPの所属部署', '担当CPの所属チーム', '「1次面接設定中」のステータス登録日',
            '面談者', '人材担当の所属Div', '人材担当の所属チーム', '企業ID', '過去経験職種',
            '求職者現在年収（単位：万円）', '面談日'
        ]

    # Drop only existing columns
    df_merged = df_merged.drop(columns=[c for c in columns_to_drop if c in df_merged.columns])

    if rename_map is None:
        rename_map = {
            '希望転職時期': '転職温度感',
            '転職の温度感': '現在住所',
            '現在年収（単位：万円）': '現在年収',
            '職種カテゴリー': '職種',
        }

    df_merged = df_merged.rename(columns=rename_map)

    # Prepare dropna subset: exclude a few date columns from dropping
    columns_to_exclude_from_dropna = ['企業書類提出日', '１次面接日', '求職者面談日時',
                                      '求人年収下限（単位：万円）', '求人年収上限（単位：万円）',
                                      '応募承諾月', '応募承諾週', 'データ登録日']

    columns_for_dropna_subset = [col for col in df_merged.columns if col not in columns_to_exclude_from_dropna]
    df_merge_diff = df_merged.dropna(subset=columns_for_dropna_subset).copy()

    return df_merge_diff


def add_time_deltas(df_merge_diff: pd.DataFrame, df_mendan: pd.DataFrame, df_oubo: pd.DataFrame):
    """Add time delta columns used in the notebook.

    Ensures necessary date columns exist by merging when missing, then computes the four time-deltas.
    """
    df_temp = df_merge_diff.copy()

    if 'データ登録日' not in df_temp.columns and 'データ登録日' in df_mendan.columns:
        df_temp = pd.merge(df_temp, df_mendan[['求職者ID', 'データ登録日']], on='求職者ID', how='left')

    if '求職者面談日時' not in df_temp.columns and '求職者面談日時' in df_mendan.columns:
        df_temp = pd.merge(df_temp, df_mendan[['求職者ID', '求職者面談日時']], on='求職者ID', how='left')

    if '応募承諾週' not in df_temp.columns and '応募承諾週' in df_oubo.columns:
        df_temp = pd.merge(df_temp, df_oubo[['求職者ID', '応募承諾週']], on='求職者ID', how='left')

    if '１次面接日' not in df_temp.columns and '１次面接日' in df_oubo.columns:
        df_temp = pd.merge(df_temp, df_oubo[['求職者ID', '１次面接日']], on='求職者ID', how='left')

    date_cols = ['データ登録日', '求職者面談日時', '応募承諾週', '１次面接日']
    for col in date_cols:
        if col in df_temp.columns:
            df_temp[col] = pd.to_datetime(df_temp[col], errors='coerce')

    df_temp['登録→面談日数'] = (df_temp['求職者面談日時'] - df_temp['データ登録日']).dt.days
    df_temp['面談→応募承諾日数'] = (df_temp['応募承諾週'] - df_temp['求職者面談日時']).dt.days
    df_temp['応募承諾→1面日数'] = (df_temp['１次面接日'] - df_temp['応募承諾週']).dt.days
    df_temp['登録→応募承諾日数'] = (df_temp['応募承諾週'] - df_temp['データ登録日']).dt.days

    time_diff_cols = ['登録→面談日数', '面談→応募承諾日数', '応募承諾→1面日数', '登録→応募承諾日数']
    for col in time_diff_cols:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].apply(lambda x: max(0, x) if pd.notna(x) else x)
            df_temp[col] = df_temp[col].fillna(0).astype(int)

    return df_temp


def finalize_dataset(df_merge_diff: pd.DataFrame, df_mendan: pd.DataFrame, df_oubo: pd.DataFrame, df_seiyaku: pd.DataFrame):
    """Create unique per 求職者ID dataset, filter and merge BID from df_seiyaku.

    Filters: 現在年収 < 600, メイン紹介経路 == 'CP厳選', エントリー数 <= 100
    """
    # First create 求職者ID-unique dataset
    df_unique = df_merge_diff.drop_duplicates(subset=['求職者ID'], keep='first').copy()

    # Filter 年収 and 紹介経路
    if '現在年収' in df_unique.columns:
        df_unique = df_unique[df_unique['現在年収'] < 600].copy()

    if 'メイン紹介経路' in df_unique.columns:
        df_unique = df_unique[df_unique['メイン紹介経路'] == 'CP厳選'].copy()

    # Cap エントリー数 <= 100 if exists
    if 'エントリー数' in df_unique.columns:
        df_unique = df_unique[df_unique['エントリー数'] <= 100].copy()

    # Merge BID
    if {'求職者ID', 'BID'}.issubset(df_seiyaku.columns):
        df_seiyaku_bid = df_seiyaku[['求職者ID', 'BID']].copy()
        df_unique = pd.merge(df_unique, df_seiyaku_bid, on='求職者ID', how='left')
        df_unique['BID'] = df_unique['BID'].fillna(0).astype(int)
        df_unique['BID'] = df_unique['BID'].apply(lambda x: 1 if x != 0 else 0)

    return df_unique
