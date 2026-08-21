"""
pages/4_Xac_Minh_Case.py — Trang 4: Xác minh Case
Vận hành chọn trạng thái xác minh cho từng entity bị flag (High/Medium):
  Có (xác nhận gian lận) | Không (báo sai) | Theo dõi (watchlist)
Lưu vào data/case_verifications.parquet — không đụng đến cases.parquet /
entity_summary_*.parquet (dữ liệu model giữ nguyên, xác minh là lớp riêng).
"""

import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Xác minh Case — XanhSM Fraud",
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
VERIF_PATH = os.path.join(DATA_DIR, "case_verifications.parquet")

STATUS_OPTIONS = ["Có", "Không", "Theo dõi"]
STATUS_ICON = {"Có": "✅", "Không": "❌", "Theo dõi": "👀"}

ENTITY_META = {
    "customer": {"id_col": "customer_id", "score_col": "fraud_score_max", "tier_col": "tier_worst", "label": "Khách hàng"},
    "driver":   {"id_col": "driver_id",   "score_col": "driver_percentile", "tier_col": "driver_tier", "label": "Tài xế"},
    "merchant": {"id_col": "merchant_id", "score_col": "fraud_score_max", "tier_col": "tier_worst", "label": "Quán ăn"},
}


# ── Data loaders ────────────────────────────────────────────────────────────────
@st.cache_data
def load_entity_summary(entity_type: str) -> pd.DataFrame:
    return pd.read_parquet(ENTITY_SUMMARY[entity_type])


@st.cache_data
def load_entity_index() -> pd.DataFrame:
    return pd.read_parquet(ENTITY_INDEX_PATH)


def load_verifications() -> pd.DataFrame:
    if os.path.exists(VERIF_PATH):
        return pd.read_parquet(VERIF_PATH)
    return pd.DataFrame(columns=[
        "entity_type", "entity_id", "verification_status", "note",
        "score_at_verification", "tier_at_verification", "verified_at",
    ])


def save_verification(entity_type, entity_id, status, note, score, tier):
    df = load_verifications()
    df = df[~((df["entity_type"] == entity_type) & (df["entity_id"] == entity_id))]
    new_row = pd.DataFrame([{
        "entity_type": entity_type,
        "entity_id": entity_id,
        "verification_status": status,
        "note": note or "",
        "score_at_verification": float(score),
        "tier_at_verification": tier,
        "verified_at": pd.Timestamp.now(),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_parquet(VERIF_PATH, index=False)


def remove_verification(entity_type, entity_id):
    df = load_verifications()
    df = df[~((df["entity_type"] == entity_type) & (df["entity_id"] == entity_id))]
    df.to_parquet(VERIF_PATH, index=False)


def build_rules_matched(entity_ids: pd.Series, entity_type: str, df_idx: pd.DataFrame) -> pd.Series:
    subset = df_idx[df_idx["entity_type"] == entity_type]
    grouped = (
        subset.groupby("entity_id")["rule_name"]
        .apply(lambda s: ", ".join(sorted(s.unique())))
        .rename("rules_matched")
    )
    return grouped


# ══════════════════════════════════════════════════════════════════════════════
# RENDER 1 TAB (1 loại entity)
# ══════════════════════════════════════════════════════════════════════════════
def render_entity_tab(entity_type: str):
    meta = ENTITY_META[entity_type]
    id_col, score_col, tier_col, label = meta["id_col"], meta["score_col"], meta["tier_col"], meta["label"]

    df_summary = load_entity_summary(entity_type)
    df_idx = load_entity_index()
    df_verif = load_verifications()
    df_verif_this = df_verif[df_verif["entity_type"] == entity_type]

    # Chỉ đưa vào hàng chờ xác minh các entity KHÔNG phải Normal
    df_queue = df_summary[df_summary[tier_col] != "Normal"].copy()

    # Gắn rules_matched
    rules_matched_ser = build_rules_matched(df_queue[id_col], entity_type, df_idx)
    df_queue = df_queue.merge(
        rules_matched_ser.reset_index().rename(columns={"entity_id": id_col}),
        on=id_col, how="left",
    )
    df_queue["rules_matched"] = df_queue["rules_matched"].fillna("—")

    # Gắn trạng thái xác minh hiện tại
    verif_map = df_verif_this.set_index("entity_id")["verification_status"].to_dict()
    df_queue["trang_thai_xac_minh"] = df_queue[id_col].map(verif_map).fillna("Chưa xác minh")

    # ── Metrics ──────────────────────────────────────────────────────────────
    n_total = len(df_queue)
    n_pending = int((df_queue["trang_thai_xac_minh"] == "Chưa xác minh").sum())
    n_co = int((df_queue["trang_thai_xac_minh"] == "Có").sum())
    n_khong = int((df_queue["trang_thai_xac_minh"] == "Không").sum())
    n_theodoi = int((df_queue["trang_thai_xac_minh"] == "Theo dõi").sum())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{label} High/Medium", f"{n_total:,}")
    m2.metric("Chờ xác minh", f"{n_pending:,}")
    m3.metric("✅ Có", f"{n_co:,}")
    m4.metric("❌ Không", f"{n_khong:,}")
    m5.metric("👀 Theo dõi", f"{n_theodoi:,}")

    # ── Bộ lọc ───────────────────────────────────────────────────────────────
    f1, f2 = st.columns(2)
    with f1:
        tier_filter = st.multiselect(
            "Lọc theo Tier:", options=sorted(df_queue[tier_col].unique().tolist()),
            default=[], key=f"verif_tier_{entity_type}",
        )
    with f2:
        status_filter = st.multiselect(
            "Lọc theo Trạng thái xác minh:",
            options=["Chưa xác minh"] + STATUS_OPTIONS,
            default=[], key=f"verif_status_{entity_type}",
        )

    df_filtered = df_queue.copy()
    if tier_filter:
        df_filtered = df_filtered[df_filtered[tier_col].isin(tier_filter)]
    if status_filter:
        df_filtered = df_filtered[df_filtered["trang_thai_xac_minh"].isin(status_filter)]

    df_filtered = df_filtered.sort_values(score_col, ascending=False).reset_index(drop=True)

    # ── Bảng xác minh hàng loạt (sửa trực tiếp, cuộn toàn bộ) ──────────────────
    st.write(
        f"**Hàng chờ xác minh — sửa trực tiếp cột \"Trạng thái\" rồi bấm Lưu "
        f"({len(df_filtered):,} / {n_total:,} {label.lower()} High+Medium):**"
    )
    if len(df_filtered) > 3000:
        st.caption(
            f"⚠️ Đang hiển thị {len(df_filtered):,} dòng — có thể hơi chậm khi cuộn. "
            f"Dùng bộ lọc Tier/Trạng thái ở trên để thu hẹp nếu cần."
        )

    editor_cols = [id_col, score_col, tier_col, "rules_matched", "trang_thai_xac_minh", "ghi_chu"]
    df_editor_src = df_filtered.copy()
    note_map = df_verif_this.set_index("entity_id")["note"].to_dict() if len(df_verif_this) else {}
    df_editor_src["ghi_chu"] = df_editor_src[id_col].map(note_map).fillna("")

    editor_key = f"editor_{entity_type}_{'-'.join(sorted(tier_filter))}_{'-'.join(sorted(status_filter))}"
    edited_df = st.data_editor(
        df_editor_src[editor_cols],
        key=editor_key,
        width="stretch",
        height=450,
        num_rows="fixed",
        hide_index=True,
        disabled=[id_col, score_col, tier_col, "rules_matched"],
        column_config={
            "trang_thai_xac_minh": st.column_config.SelectboxColumn(
                "Trạng thái", options=["Chưa xác minh"] + STATUS_OPTIONS, required=True,
            ),
            "ghi_chu": st.column_config.TextColumn("Ghi chú"),
            score_col: st.column_config.NumberColumn(score_col, format="%.4f"),
        },
    )

    if st.button("💾 Lưu tất cả thay đổi", key=f"bulk_save_{entity_type}", type="primary"):
        n_changed = 0
        original = df_editor_src.set_index(id_col)
        for _, row in edited_df.iterrows():
            eid = row[id_col]
            new_status = row["trang_thai_xac_minh"]
            new_note = row["ghi_chu"] or ""
            old_status = original.loc[eid, "trang_thai_xac_minh"]
            old_note = original.loc[eid, "ghi_chu"] or ""
            if new_status == old_status and new_note == old_note:
                continue
            if new_status == "Chưa xác minh":
                remove_verification(entity_type, eid)
            else:
                save_verification(
                    entity_type, eid, new_status, new_note,
                    row[score_col], row[tier_col],
                )
            n_changed += 1
        if n_changed:
            st.success(f"Đã lưu {n_changed:,} thay đổi.")
            st.rerun()
        else:
            st.info("Không có thay đổi nào để lưu.")

    st.divider()

    # ── Khu vực xác minh 1 entity cụ thể (tra cứu nhanh, xem chi tiết) ────────
    st.write("**Tra cứu 1 case cụ thể (xem chi tiết đầy đủ / ghi chú riêng):**")
    lookup_input = st.text_input(
        f"Nhập {id_col} cần xác minh:",
        key=f"verif_lookup_{entity_type}",
        placeholder="Ví dụ: e1bccf5c09705f63",
    )

    picked_id = lookup_input.strip() if lookup_input else None

    if st.session_state.get(f"verif_active_{entity_type}"):
        picked_id = st.session_state[f"verif_active_{entity_type}"]

    if picked_id:
        match = df_summary[df_summary[id_col] == picked_id]
        if len(match) == 0:
            st.warning(f"Không tìm thấy `{picked_id}` trong entity_summary_{entity_type}.parquet.")
        else:
            r = match.iloc[0]
            cur_status = verif_map.get(picked_id, "Chưa xác minh")
            cur_note = ""
            existing = df_verif_this[df_verif_this["entity_id"] == picked_id]
            if len(existing):
                cur_note = existing.iloc[0].get("note", "") or ""

            st.info(
                f"**Đang xác minh:** `{picked_id}`  |  {score_col}: **{r[score_col]:.4f}**  |  "
                f"Tier: **{r[tier_col]}**  |  Trạng thái hiện tại: **{cur_status}**"
            )

            col_link, _ = st.columns([2, 5])
            with col_link:
                if st.button("Xem Chi tiết đầy đủ (Trang 3)", key=f"goto3_{entity_type}_{picked_id}"):
                    st.session_state["selected_entity_type"] = entity_type
                    st.session_state["selected_entity_id"] = picked_id
                    st.switch_page("pages/3_Chi_tiet_Entity.py")

            note_input = st.text_input(
                "Ghi chú (tuỳ chọn):", value=cur_note, key=f"note_{entity_type}_{picked_id}",
            )

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("✅ Có (xác nhận gian lận)", key=f"btn_co_{entity_type}_{picked_id}", type="primary"):
                    save_verification(entity_type, picked_id, "Có", note_input, r[score_col], r[tier_col])
                    st.success(f"Đã xác nhận **Có** cho `{picked_id}`.")
                    st.rerun()
            with b2:
                if st.button("❌ Không (báo sai)", key=f"btn_khong_{entity_type}_{picked_id}"):
                    save_verification(entity_type, picked_id, "Không", note_input, r[score_col], r[tier_col])
                    st.success(f"Đã đánh dấu **Không** cho `{picked_id}`.")
                    st.rerun()
            with b3:
                if st.button("👀 Theo dõi", key=f"btn_theodoi_{entity_type}_{picked_id}"):
                    save_verification(entity_type, picked_id, "Theo dõi", note_input, r[score_col], r[tier_col])
                    st.success(f"Đã đưa `{picked_id}` vào **Theo dõi**.")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.title("Xác minh Case")
st.caption(
    "Vận hành xác minh từng entity bị flag (Tier High/Medium): Có / Không / Theo dõi. "
    "Kết quả lưu tại `data/case_verifications.parquet` — không thay đổi fraud_score/tier gốc."
)

df_verif_all = load_verifications()
if len(df_verif_all):
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Tổng đã xác minh (mọi loại)", f"{len(df_verif_all):,}")
    t2.metric("✅ Có", f"{int((df_verif_all['verification_status'] == 'Có').sum()):,}")
    t3.metric("❌ Không", f"{int((df_verif_all['verification_status'] == 'Không').sum()):,}")
    t4.metric("👀 Theo dõi", f"{int((df_verif_all['verification_status'] == 'Theo dõi').sum()):,}")
else:
    st.info("Chưa có case nào được xác minh.")

st.divider()

tab_customer, tab_driver, tab_merchant = st.tabs(["Khách hàng", "Tài xế", "Quán ăn"])

with tab_customer:
    render_entity_tab("customer")

with tab_driver:
    render_entity_tab("driver")

with tab_merchant:
    render_entity_tab("merchant")
