"""
pages/2_Detection_Results.py — Trang 2: Kết quả Phát hiện Gian lận
3 tab: Khách hàng | Tài xế | Quán ăn
Logic lọc: OR/UNION theo rule_name từ rule_entity_index
"""

import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Kết quả Phát hiện Gian lận — XanhSM Fraud",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(APP_DIR, "data")

ENTITY_SUMMARY = {
    "customer": os.path.join(DATA_DIR, "entity_summary_customer.parquet"),
    "merchant": os.path.join(DATA_DIR, "entity_summary_merchant.parquet"),
    "driver":   os.path.join(DATA_DIR, "entity_summary_driver.parquet"),
}
ENTITY_INDEX_PATH = os.path.join(DATA_DIR, "rule_entity_index.parquet")

# ── Data loaders (cached) ──────────────────────────────────────────────────────
@st.cache_data
def load_entity_summary(entity_type: str) -> pd.DataFrame:
    return pd.read_parquet(ENTITY_SUMMARY[entity_type])

@st.cache_data
def load_entity_index() -> pd.DataFrame:
    return pd.read_parquet(ENTITY_INDEX_PATH)

def build_rules_matched(entity_ids: pd.Series, entity_type: str, df_idx: pd.DataFrame) -> pd.Series:
    """Với mỗi entity_id, nối tất cả rule_name từng dính (không chỉ rule đang bật)."""
    subset = df_idx[df_idx["entity_type"] == entity_type]
    grouped = (
        subset.groupby("entity_id")["rule_name"]
        .apply(lambda s: ", ".join(sorted(s.unique())))
        .rename("rules_matched")
    )
    return grouped


def render_entity_tab(
    entity_type: str,      # "customer" | "merchant" | "driver"
    id_col: str,           # "customer_id" | "merchant_id" | "driver_id"
    score_col: str,        # "fraud_score_max" | "driver_percentile"
    tier_col: str,         # "tier_worst" | "driver_tier"
    tab_label: str,        # tên hiển thị để dùng trong st.info
):
    df_summary = load_entity_summary(entity_type)
    df_idx = load_entity_index()

    # ── Tập rule có thực sự xuất hiện cho entity_type này ──────────────────────
    idx_subset = df_idx[df_idx["entity_type"] == entity_type]
    available_rules = sorted(idx_subset["rule_name"].unique().tolist())

    # ── Bộ lọc Rule (OR logic) ─────────────────────────────────────────────────
    selected_rules = st.multiselect(
        f"Lọc theo rule vi phạm ({tab_label}) — bật nhiều rule = OR/UNION:",
        options=available_rules,
        default=[],
        key=f"multiselect_{entity_type}",
    )

    # ── Áp dụng lọc ───────────────────────────────────────────────────────────
    total_all = len(df_summary)
    if selected_rules:
        flagged_ids = idx_subset.loc[
            idx_subset["rule_name"].isin(selected_rules), "entity_id"
        ].unique()
        df_filtered = df_summary[df_summary[id_col].isin(flagged_ids)].copy()
    else:
        df_filtered = df_summary.copy()

    total_shown = len(df_filtered)

    # ── Metrics ────────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số entity trong hệ thống", f"{total_all:,}")
    m2.metric("Đang hiển thị (sau lọc)" if selected_rules else "Đang hiển thị (toàn bộ)", f"{total_shown:,}")
    if selected_rules:
        m3.metric("Tỷ lệ bị lọc", f"{total_shown/total_all*100:.2f}%")

    # ── Gắn cột rules_matched ──────────────────────────────────────────────────
    rules_matched_ser = build_rules_matched(df_filtered[id_col], entity_type, df_idx)
    df_filtered = df_filtered.merge(
        rules_matched_ser.reset_index().rename(columns={"entity_id": id_col}),
        on=id_col,
        how="left",
    )
    df_filtered["rules_matched"] = df_filtered["rules_matched"].fillna("—")

    # ── Sort theo score giảm dần ───────────────────────────────────────────────
    df_filtered = df_filtered.sort_values(score_col, ascending=False).reset_index(drop=True)

    # ── Hiển thị bảng + nút Chi tiết ──────────────────────────────────────────
    st.write(f"**Danh sách {tab_label}** ({'lọc theo rule đã chọn' if selected_rules else 'toàn bộ, chưa lọc'}):")

    # Hiện tối đa 200 dòng để giao diện không chậm
    display_df = df_filtered.head(200)

    # Hiển thị bảng (không nút inline vì st.dataframe không hỗ trợ nút trong ô)
    st.dataframe(display_df, width='stretch', height=400)

    if total_shown > 200:
        st.caption(f"⚠️ Chỉ hiển thị 200/{total_shown:,} dòng đầu (sort theo {score_col} giảm dần).")

    # ── Tra cứu nhanh entity_id cụ thể ────────────────────────────────────────
    st.write("**Tra cứu nhanh / Chọn xem Chi tiết:**")
    lookup_input = st.text_input(
        f"Nhập {id_col} để xem chi tiết:",
        key=f"lookup_{entity_type}",
        placeholder=f"Ví dụ: e1bccf5c09705f63",
    )

    if lookup_input:
        match = df_filtered[df_filtered[id_col] == lookup_input.strip()]
        if len(match):
            r = match.iloc[0]
            st.dataframe(match, width='stretch')
            if st.button(f"Chọn {lookup_input.strip()} — Xem Chi tiết", key=f"btn_lookup_{entity_type}"):
                st.session_state["selected_entity_type"] = entity_type
                st.session_state["selected_entity_id"] = lookup_input.strip()
                st.switch_page("pages/3_Chi_tiet_Entity.py")
        else:
            st.warning(f"Không tìm thấy `{lookup_input.strip()}` trong danh sách đang hiển thị.")

    # ── Nút chọn nhanh từ Top 10 ──────────────────────────────────────────────
    st.write("**Chọn nhanh từ Top 10 rủi ro cao nhất:**")
    top10 = df_filtered.head(10)
    for _, row in top10.iterrows():
        eid = row[id_col]
        score_val = row[score_col]
        tier_val = row[tier_col]
        rules_val = row.get("rules_matched", "—")
        col_a, col_b = st.columns([5, 1])
        with col_a:
            st.write(
                f"`{eid}` | {score_col}: **{score_val:.4f}** | Tier: **{tier_val}** | Rules: {rules_val}"
            )
        with col_b:
            if st.button("Chi tiết", key=f"btn_{entity_type}_{eid}"):
                st.session_state["selected_entity_type"] = entity_type
                st.session_state["selected_entity_id"] = eid
                st.switch_page("pages/3_Chi_tiet_Entity.py")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.title("Kết quả Phát hiện Gian lận")
st.caption(
    "Dữ liệu từ: `cases.parquet` (Isolation Forest), `driver_scores.parquet`, `rule_entity_index.parquet`. "
    "Lọc rule dùng logic OR: một entity xuất hiện trong BẤT KỲ rule nào đang chọn sẽ được giữ lại."
)

# Hiển thị entity đang được chọn (nếu có)
if st.session_state.get("selected_entity_id"):
    sel_type = st.session_state.get("selected_entity_type", "?")
    sel_id   = st.session_state.get("selected_entity_id", "?")
    st.info(f"🔍 Entity đang chọn: **{sel_type}** / `{sel_id}`")

tab_customer, tab_driver, tab_merchant = st.tabs(["Khách hàng", "Tài xế", "Quán ăn"])

with tab_customer:
    render_entity_tab(
        entity_type="customer",
        id_col="customer_id",
        score_col="fraud_score_max",
        tier_col="tier_worst",
        tab_label="Khách hàng",
    )

with tab_driver:
    render_entity_tab(
        entity_type="driver",
        id_col="driver_id",
        score_col="driver_percentile",
        tier_col="driver_tier",
        tab_label="Tài xế",
    )

with tab_merchant:
    render_entity_tab(
        entity_type="merchant",
        id_col="merchant_id",
        score_col="fraud_score_max",
        tier_col="tier_worst",
        tab_label="Quán ăn",
    )
