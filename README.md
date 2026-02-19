# AI↔他分野 推薦アプリ（E5固定 / データ差し替え対応）

- デフォルトでは `data/` 配下の JSONL / CSV を使用します。
- 必要ならサイドバーから **JSONL / アンケートCSV** をアップロードして差し替えできます。
  - JSONL: `.jsonl`
  - CSV: `id,url` 列がある想定（URL列が無い場合は空になります）

E5の構造（query:/passage:、normalize_embeddings=True）は変更していません。

## 起動
```bash
pip install -r requirements.txt
streamlit run app.py
```
