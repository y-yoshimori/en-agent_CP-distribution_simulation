# エンエージェント：CP配布〜入社決定データ分析ノートブック

このリポジトリは、担当CP（キャリアパートナー）ごとの配布から入社決定に至る分析を行うためのJupyter Notebookと、解析パイプラインをモジュール化したPythonスクリプト群を含みます。ノートブックはデータ読み込み → 前処理 → 特徴量エンジニアリング → 統計検定（マン・ホイットニーU検定）→ モデル学習（LightGBM）→ ハイパーパラメータ最適化（Optuna）→ 解釈（Feature importance / SHAP）という流れで構成されています。

## 目次
- リポジトリ構成
- ノートブックの目的
- ノートブックで実行する主な処理（概要）
- モジュール（`scripts/`）の説明
- 実行方法（推奨セル実行順）
- 実行上の注意点
- 今後の改善案

## リポジトリ構成（主要ファイル）
- `en-agent_CP-distribution_analytics.ipynb` — メインの解析ノートブック（前処理〜可視化まで）。
- `scripts/preprocessing.py` — データ結合・欠損処理・日付差分などの前処理関数。
- `scripts/feature_engineering.py` — 求職者単位の集計や新しい特徴量（年収ギャップ、転職経験、紹介経路集計など）。
- `scripts/u_test.py` — 各CP・各特徴量ごとにマン・ホイットニーU検定を実行する関数。
- `scripts/model.py` — LightGBM 用のデータ準備（エンコーディング・欠損処理）と CV 学習/評価関数。
- `scripts/optuna_utils.py` — Optuna を用いた LightGBM のハイパーパラメータ探索ラッパー。
- `scripts/utils.py` — CP 単位でデータ分割するユーティリティ等。
- `scripts/interpret.py` — 特徴量重要度の描画と SHAP 値の計算／描画ヘルパー。

## ノートブックの目的
- 各担当CPが「どのような求職者属性に強いか」を定量的に把握する。
- 属性ごとにエントリー数（≒入社決定の前段階）に差が生じる場合を“CPの強み”と定義し、統計検定で検出する。
- 強みのある属性をもつ CP に配布したときの効果（リフト）を推定・シミュレーションするための下地を作る。

## ノートブックで実行する主な処理（概要）
1. 必要ライブラリの一括インストール（ノートブック冒頭に pip インストールセルあり）。
2. データ読み込み：面談シート、応募シート、成約シートなどを読み込む。
3. 前処理：不要列除去、結合（応募 ↔ 面談）、欠損行の扱い、日付差分（登録→面談、面談→応募承諾 等）など。
4. 特徴量エンジニアリング：転職経験フラグ、ランクギャップ、登録経路（大分類）、年収ギャップ、求職者ごとのエントリー数や業種/職種一致率、平均年収ギャップ、メイン紹介経路などを作成。
5. 求職者単位にユニーク化（1人1行）し、フィルタ（年収上限、エントリー数上限など）を適用。
6. 担当CPごとにデータ分割し、条件に合う CP を抽出（例：10 <= rows < 200 など）。
7. 各 CP・各特徴量のカテゴリについて、該当カテゴリのエントリー数とその他のカテゴリを比較して Mann–Whitney U 検定を実施。p 値や効果量を集計。
8. 重要な特徴（p < 0.05 かつ効果量の大きいもの）を一覧化・可視化。
9. 予測モデル（LightGBM）で入社決定（BID）を予測するための学習パイプライン。クロスバリデーションで AUC/F1 を評価。学習ループは `scripts.model.train_lgb_cv` で提供。
10. Optuna によるハイパーパラメータ探索（`scripts.optuna_utils.optuna_search`）。
11. Feature importance の可視化と、SHAP による特徴量寄与の解釈（`scripts.interpret` を利用）。

## 実行方法（推奨：ノートブック内のセル順で実行）
ノートブックはセル単位で実行するスタイルを想定しています。推奨の実行順は以下です：

1. 先頭の「必要ライブラリ一括インストール」セルを実行（1回だけ）。

	PowerShell 例（手動でインストールする場合）:
```powershell
python -m pip install pandas numpy matplotlib seaborn japanize-matplotlib lightgbm scikit-learn optuna shap
```

2. データ読み込みセルを実行して、`df_mendan`, `df_oubo`, `df_seiyaku` を作成。
3. 「パイプライン実行」セルを実行（内部で `scripts/preprocessing.py` → `scripts/feature_engineering.py` → `scripts/utils.split_by_cp` → `scripts/u_test.run_mannwhitney_tests` を呼びます）。これによって `df_final`, `filtered_cp_dataframes_list`, `statistical_analysis_df` 等が生成されます。
4. EDA（可視化）セルを順に実行して、重要な属性・カテゴリを確認。
5. モデル学習や Optuna を試す場合：
	- ノートブック内では学習／Optuna 呼び出しはデフォルトでコメント化されています。実行する場合は該当セルのコメントを外して実行してください。
	- 最初は軽めの設定（`n_trials=10`、`n_estimators` 少なめ）で動作確認することを推奨します。
6. 学習済みモデルが得られたら、`scripts/interpret.py` の関数（`plot_feature_importance`, `compute_shap_values`, `plot_shap_summary`, `plot_shap_dependence`）を使って解釈プロットを作成します。

## 実行上の注意点
- LightGBM や SHAP、Optuna は計算負荷が高い場合があります。最初は少数の試行で動作確認してください。Optuna は探索回数（n_trials）を増やすほど時間がかかります。
- Windows 環境では LightGBM のインストールにビルド依存があり、環境によって追加手順が必要になる場合があります（Anaconda などを利用するケースが多い）。
- ノートブックは「セルを順に実行」することを前提にしています。未実行のセルに依存するコードがあるため、順序を変更するとエラーが出ることがあります。

## スクリプト（`scripts/`）について
- 既存のノートブック内の大きなコードブロックは、下記のモジュールに切り出されています。コードを繰り返すことなく関数呼び出しで実行できるようになっています。
  - `scripts/preprocessing.py`：merge、不要列削除、日付差分（登録→面談 等）の作成
  - `scripts/feature_engineering.py`：転職経験、ランクギャップ、年収ギャップ、エントリー一致率等の計算
  - `scripts/u_test.py`：Mann–Whitney U 検定を CP ごとに実行して p 値・効果量を返す
  - `scripts/model.py`：LightGBM 用の前処理（ラベルエンコード/数値化）と CV 学習関数
  - `scripts/optuna_utils.py`：Optuna の探索ラッパー（CV 内での評価を行います）
  - `scripts/interpret.py`：特徴量重要度・SHAP を計算・プロットするユーティリティ
  - `scripts/utils.py`：CP 分割や共通ユーティリティ

## 開発上のメモ / 今後の改善案
- `requirements.txt` を固定したい場合は自動生成します（希望があれば作成します）。
- 主要な関数に対する小さなユニットテスト（sanity check）を追加すると、将来的な変更での回帰を防げます。
- 現在、学習／Optuna はノートブックでオプション実行となっています。CI での軽い smoke-test（例：n_trials=2）を用意すると自動検証しやすくなります。

---

この README はノートブックのマークダウン構成を踏まえてまとめたものです。ノートブックを実行して出たエラーや気になる点があれば教えてください。README の補足（依存の固定化、実行例のスクリーンショット追加、詳細な API ドキュメント生成など）も対応します。
