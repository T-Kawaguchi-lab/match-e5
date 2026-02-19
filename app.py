import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ------------------------
# Fixed settings (keep E5 structure unchanged)
# ------------------------
DEFAULT_MODEL = "intfloat/multilingual-e5-base"
ROLE_PATH = "meta.role"

TEXT_KEY_PRIORITY = [
    "match_text.canonical_card_text",
    "e5_text",
    "e5_passage",
    "e5_query",
]

st.set_page_config(page_title="AI↔他分野 推薦（E5）", layout="wide")
st.title("AI研究者 ↔ 他分野研究者 推薦（E5 cosine）")
st.caption("E5（query:/passage:）+ normalize_embeddings=True を使用して類似度を計算します。")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"


def read_jsonl_from_path(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_jsonl_from_uploaded(uploaded) -> List[Dict[str, Any]]:
    content = uploaded.getvalue().decode("utf-8", errors="ignore").splitlines()
    return [json.loads(line) for line in content if line.strip()]


def read_csv_from_path(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_csv_from_uploaded(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def get_nested(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def normalize_role_value(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    if s in {"ai_researcher", "ai", "provider", "system_researcher", "system", "ai_research"}:
        return "ai_researcher"
    if s in {"other_field_researcher", "other", "needs", "science_researcher", "domain_researcher", "non_ai", "other_field"}:
        return "other_field_researcher"
    return s


def ensure_prefix(text: str, prefix: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if re.match(r"^\s*(query:|passage:)\s*", t, flags=re.IGNORECASE):
        t = re.sub(r"^\s*(query:|passage:)\s*", prefix + " ", t, flags=re.IGNORECASE)
        return t.strip()
    return f"{prefix} {t}".strip()


def summarize_one_line(r: Dict[str, Any]) -> str:
    v = get_nested(r, "match_text.one_line_pitch")
    if isinstance(v, str) and v.strip():
        return v.strip()
    v = get_nested(r, "match_text.canonical_card_text")
    if isinstance(v, str) and v.strip():
        return (v.strip()[:160] + "…") if len(v.strip()) > 160 else v.strip()
    return ""


def get_text_by_priority(r: Dict[str, Any], priorities: List[str]) -> str:
    for key in priorities:
        v = get_nested(r, key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = get_nested(r, "match_text.canonical_card_text")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return json.dumps(r.get("meta", {}), ensure_ascii=False)


def build_id(i_1based: int) -> str:
    return f"R{i_1based:04d}"


@st.cache_resource
def load_model(model_name: str):
    return SentenceTransformer(model_name)


def embed_e5(model: SentenceTransformer, query_text: str, doc_texts: List[str]) -> np.ndarray:
    q = model.encode([ensure_prefix(query_text, "query:")], normalize_embeddings=True, show_progress_bar=False)
    d = model.encode([ensure_prefix(t, "passage:") for t in doc_texts], normalize_embeddings=True, show_progress_bar=False)
    q = np.asarray(q, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    return (q @ d.T).reshape(-1)


# ------------------------
# Data selection UI (defaults to current files)
# ------------------------
with st.sidebar:
    st.header("データ選択（任意）")
    st.caption("デフォルトはリポジトリ内の data/ を使用します。必要ならここで差し替えできます。")

    # JSONL
    jsonl_files = sorted([p.name for p in DATA_DIR.glob("*.jsonl")])
    default_jsonl = jsonl_files[0] if jsonl_files else None

    jsonl_mode = st.radio("JSONLの読み込み", ["既存ファイルを使う", "アップロードして差し替える"], index=0)
    selected_jsonl_name = None
    uploaded_jsonl = None
    if jsonl_mode == "既存ファイルを使う":
        if default_jsonl is None:
            st.error("data/ に JSONL がありません。アップロードしてください。")
        else:
            selected_jsonl_name = st.selectbox("JSONLファイル", jsonl_files, index=0)
    else:
        uploaded_jsonl = st.file_uploader("JSONLをアップロード", type=["jsonl"])

    st.divider()

    # CSV
    csv_files = sorted([p.name for p in DATA_DIR.glob("*.csv")])
    default_csv = "url_mapping_mock.csv" if "url_mapping_mock.csv" in csv_files else (csv_files[0] if csv_files else None)

    csv_mode = st.radio("アンケートCSVの読み込み", ["既存ファイルを使う", "アップロードして差し替える"], index=0)
    selected_csv_name = None
    uploaded_csv = None
    if csv_mode == "既存ファイルを使う":
        if default_csv is None:
            st.warning("data/ に CSV がありません（URL列は空になります）。必要ならアップロードしてください。")
        else:
            idx = csv_files.index(default_csv) if default_csv in csv_files else 0
            selected_csv_name = st.selectbox("CSVファイル", csv_files, index=idx)
    else:
        uploaded_csv = st.file_uploader("CSVをアップロード（id,url列がある想定）", type=["csv"])

    st.divider()
    st.caption(f"使用モデル: {DEFAULT_MODEL}")


# ------------------------
# Load selected data
# ------------------------
if jsonl_mode == "アップロードして差し替える":
    if uploaded_jsonl is None:
        st.warning("JSONLが未指定です。サイドバーでアップロードしてください。")
        st.stop()
    rows = read_jsonl_from_uploaded(uploaded_jsonl)
    jsonl_label = f"uploaded:{uploaded_jsonl.name}"
else:
    if selected_jsonl_name is None:
        st.error("JSONLが見つかりません。data/に置くか、アップロードしてください。")
        st.stop()
    rows = read_jsonl_from_path(DATA_DIR / selected_jsonl_name)
    jsonl_label = selected_jsonl_name

if not rows:
    st.error("JSONLが空です。")
    st.stop()

if csv_mode == "アップロードして差し替える":
    if uploaded_csv is None:
        map_df = pd.DataFrame(columns=["id", "url"])
        csv_label = "(none)"
    else:
        map_df = read_csv_from_uploaded(uploaded_csv)
        csv_label = f"uploaded:{uploaded_csv.name}"
else:
    if selected_csv_name is None:
        map_df = pd.DataFrame(columns=["id", "url"])
        csv_label = "(none)"
    else:
        map_df = read_csv_from_path(DATA_DIR / selected_csv_name)
        csv_label = selected_csv_name

st.caption(f"データ: JSONL={jsonl_label} / CSV={csv_label}")


# ------------------------
# Build df
# ------------------------
records = []
roles_raw = []
for i, r in enumerate(rows, start=1):
    rid = build_id(i)
    meta = r.get("meta", {}) if isinstance(r.get("meta", {}), dict) else {}
    role_raw = get_nested(r, ROLE_PATH)
    role_n = normalize_role_value(role_raw)
    roles_raw.append(role_raw)

    records.append({
        "id": rid,
        "role_norm": role_n,
        "name": meta.get("name") or meta.get("name_raw") or "",
        "affiliation": meta.get("affiliation") or "",
        "position": meta.get("position") or "",
        "research_field": meta.get("research_field") or "",
        "summary": summarize_one_line(r),
        "embed_text": get_text_by_priority(r, TEXT_KEY_PRIORITY),
    })

df = pd.DataFrame(records)

if not map_df.empty and "id" in map_df.columns and "url" in map_df.columns:
    df = df.merge(map_df[["id", "url"]], on="id", how="left")
else:
    df["url"] = ""

ai_df = df[df["role_norm"] == "ai_researcher"].reset_index(drop=True)
other_df = df[df["role_norm"] == "other_field_researcher"].reset_index(drop=True)

c1, c2, c3 = st.columns(3)
c1.metric("総件数", len(df))
c2.metric("AI研究者", len(ai_df))
c3.metric("他分野研究者", len(other_df))

if len(ai_df) == 0 or len(other_df) == 0:
    st.warning("role分離の結果、片側が0件です。meta.role の値（表記ゆれ）を確認してください。")
    st.write("role_rawのユニーク（先頭30）:", sorted({str(v) for v in roles_raw if v is not None})[:30])
    st.stop()

# ------------------------
# Minimal UI (direction + query person + topK)
# ------------------------
st.write("### 設定")
direction = st.radio("検索方向", ["AI研究者 → 他分野研究者", "他分野研究者 → AI研究者"], index=0, horizontal=True)
top_k = st.slider("Top-K", 3, 100, 20, 1)

if direction == "AI研究者 → 他分野研究者":
    query_df = ai_df
    doc_df = other_df
    query_label = "AI研究者（query）"
    doc_label = "他分野研究者（推薦先）"
else:
    query_df = other_df
    doc_df = ai_df
    query_label = "他分野研究者（query）"
    doc_label = "AI研究者（推薦先）"

st.write("### クエリ人物を選択")
labels = query_df.apply(lambda r: f'{r["name"]} | {r["affiliation"]} | {r["position"]} | {r["research_field"]}', axis=1).tolist()
sel = st.selectbox(query_label, labels, index=0)
sel_idx = labels.index(sel)
query_text = query_df.loc[sel_idx, "embed_text"]

run = st.button(f"検索実行：{query_label} → {doc_label}", type="primary")
if not run:
    st.stop()

with st.spinner("E5で埋め込み計算中..."):
    model = load_model(DEFAULT_MODEL)
    sims = embed_e5(model, query_text, doc_df["embed_text"].tolist())

top_k = min(top_k, len(doc_df))
top_idx = np.argsort(-sims)[:top_k]

res = doc_df.iloc[top_idx].copy()
res.insert(0, "rank", np.arange(1, len(res) + 1))
res.insert(1, "similarity", sims[top_idx].astype(float))

show_cols = ["rank", "similarity", "name", "affiliation", "position", "research_field", "summary", "url"]
res_show = res[show_cols].copy()

st.subheader("検索結果")
st.caption(f"使用モデル: {DEFAULT_MODEL}（E5 / query:/passage: / normalize_embeddings=True）")

try:
    st.dataframe(
        res_show,
        use_container_width=True,
        column_config={
            "url": st.column_config.LinkColumn("アンケートURL", display_text="open"),
            "similarity": st.column_config.NumberColumn("類似度", format="%.4f"),
            "rank": st.column_config.NumberColumn("順位"),
        },
        hide_index=True,
    )
except Exception:
    st.dataframe(res_show, use_container_width=True, hide_index=True)

csv_bytes = res_show.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button("結果をCSVでダウンロード（URL含む）", data=csv_bytes, file_name="match_results.csv", mime="text/csv")

json_bytes = res_show.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")
st.download_button("結果をJSONでダウンロード（URL含む）", data=json_bytes, file_name="match_results.json", mime="application/json")
