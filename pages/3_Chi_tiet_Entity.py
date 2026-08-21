"""
pages/3_Chi_tiet_Entity.py — Trang 3: Chi tiết Entity
4 tab: Rule-based | Graph | Anomaly | Lịch sử đơn hàng
Chỉ dùng component mặc định Streamlit — không CSS/HTML tùy biến.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import streamlit as st

st.set_page_config(
    page_title="Chi tiết Entity — XanhSM Fraud",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(APP_DIR, "data")
RULES_DIR = os.path.join(DATA_DIR, "rules")

# ── Paths ──────────────────────────────────────────────────────────────────────
ENTITY_INDEX_PATH   = os.path.join(DATA_DIR, "rule_entity_index.parquet")
GRAPH_EDGES_PATH    = os.path.join(DATA_DIR, "graph_edges.parquet")
GRAPH_COMM_PATH     = os.path.join(DATA_DIR, "graph_communities.parquet")
CASES_FEAT_PATH     = os.path.join(DATA_DIR, "cases_with_features.parquet")
DRIVER_SCORES_PATH  = os.path.join(DATA_DIR, "driver_scores.parquet")
ORDERS_FULL_PATH    = os.path.join(DATA_DIR, "orders_full.parquet")

# ── Quy tắc: mỗi rule file → (tên hiển thị, cột customer, cột driver, cột merchant) ──
# None = rule không dùng loại entity đó
RULE_META = {
    "kb1_trio_collusion.parquet":  ("KB-1 Trio Collusion",            "customer_id", "driver_id",  "merchant_id"),
    "kb3_merchant_driver.parquet": ("KB-3 Merchant-Driver cố định",   None,          "driver_id",  "merchant_id"),
    "kb4_shell_merchant.parquet":  ("KB-4 Shell Merchant",            "customer_id", None,         "merchant_id"),
    "kb5_fake_prep_time.parquet":  ("KB-5 Fake Prep Time",            None,          None,         "merchant_id"),
    "kb6_overlap.parquet":         ("KB-Overlap",                     None,          "driver_id",  None),
    "kb10_driver_targeted.parquet":("KB-10 Driver nhắm hại khách",    "customer_id", "driver_id",  None),
    "kb_rating.parquet":           ("KB-Rating",                      "customer_id", None,         None),
    "split_orders.parquet":        ("Split Orders",                    "customer_id", None,         "merchant_id"),
    "ghost_delivery.parquet":      ("Ghost Delivery",                  "customer_id", "driver_id",  "merchant_id"),
    "bom_hang.parquet":            ("Bom hàng (COD No-show)",          "customer_id", "driver_id",  "merchant_id"),
}

# Rules có badge "Độ tin cậy: Thấp" — từ is_low_confidence trong rule_entity_index
LOW_CONFIDENCE_RULES = {"KB-10 Driver nhắm hại khách", "KB-Rating"}

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_entity_index():
    return pd.read_parquet(ENTITY_INDEX_PATH)

@st.cache_data
def load_graph_edges():
    return pd.read_parquet(GRAPH_EDGES_PATH)

@st.cache_data
def load_graph_communities():
    return pd.read_parquet(GRAPH_COMM_PATH)

@st.cache_data
def load_cases_with_features():
    return pd.read_parquet(CASES_FEAT_PATH)

@st.cache_data
def load_driver_scores():
    return pd.read_parquet(DRIVER_SCORES_PATH)

@st.cache_data
def load_orders():
    return pd.read_parquet(ORDERS_FULL_PATH)

@st.cache_data
def load_rule_file(fname):
    return pd.read_parquet(os.path.join(RULES_DIR, fname))


# ══════════════════════════════════════════════════════════════════════════════
# ĐẦU TRANG: đọc entity từ session_state
# ══════════════════════════════════════════════════════════════════════════════
st.title("Chi tiết Entity")

entity_type = st.session_state.get("selected_entity_type")
entity_id   = st.session_state.get("selected_entity_id")

if not entity_type or not entity_id:
    st.warning("⚠️ Vui lòng chọn 1 entity từ Trang 2 trước.")
    st.stop()

# Map entity_type → cột ID tương ứng trong các file
ID_COL_MAP = {
    "customer": "customer_id",
    "driver":   "driver_id",
    "merchant": "merchant_id",
}
id_col = ID_COL_MAP[entity_type]

st.caption(f"Đang xem: **{entity_type.upper()}** / `{entity_id}`")
st.divider()

# ── 4 Tab ─────────────────────────────────────────────────────────────────────
tab_rule, tab_graph, tab_anomaly, tab_orders = st.tabs([
    "Rule-based",
    "Graph",
    "Anomaly",
    "Lịch sử đơn hàng",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: RULE-BASED
# ══════════════════════════════════════════════════════════════════════════════
with tab_rule:
    st.subheader("Vi phạm Rule-based")

    df_idx = load_entity_index()
    # Kiểm tra is_low_confidence từ rule_entity_index
    idx_entity = df_idx[
        (df_idx["entity_type"] == entity_type) &
        (df_idx["entity_id"] == entity_id)
    ]

    found_any = False

    for fname, (rname, ccol, dcol, mcol) in RULE_META.items():
        # Xác định cột dùng để tìm entity_id trong file rule này
        col_for_type = {
            "customer": ccol,
            "driver":   dcol,
            "merchant": mcol,
        }.get(entity_type)

        if col_for_type is None:
            # Rule này không áp dụng cho entity_type đang xem
            continue

        try:
            df_rule = load_rule_file(fname)
        except Exception as e:
            st.error(f"Lỗi đọc {fname}: {e}")
            continue

        if col_for_type not in df_rule.columns:
            continue

        matched = df_rule[df_rule[col_for_type] == entity_id]
        if len(matched) == 0:
            continue

        found_any = True

        # Badge độ tin cậy (dùng is_low_confidence từ rule_entity_index)
        is_low_conf = False
        if rname in LOW_CONFIDENCE_RULES:
            row_idx = idx_entity[idx_entity["rule_name"] == rname]
            if len(row_idx):
                is_low_conf = bool(row_idx.iloc[0]["is_low_confidence"])

        if is_low_conf:
            st.warning(f"**{rname}** — ⚠️ Độ tin cậy: Thấp (n<15)  |  {len(matched)} dòng vi phạm")
        else:
            st.write(f"**{rname}**  |  {len(matched)} dòng vi phạm")

        # Hiển thị đầy đủ cột — không rút gọn
        st.dataframe(matched.reset_index(drop=True), width="stretch")
        st.divider()

    if not found_any:
        st.info("Không dính rule nào.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: GRAPH
# ══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.subheader("Phân tích đồ thị cộng đồng (Louvain)")

    # Prefix của node trong graph
    PREFIX = {"customer": "C_", "driver": "D_", "merchant": "M_"}
    node_id_in_graph = PREFIX[entity_type] + entity_id

    df_comm = load_graph_communities()

    row_comm = df_comm[df_comm["node_id"] == node_id_in_graph]

    if len(row_comm) == 0:
        st.info(
            f"Entity này (`{node_id_in_graph}`) không xuất hiện trong đồ thị "
            f"(tất cả cạnh weight < 5 — không đủ ngưỡng)."
        )
        st.stop()

    in_flagged = bool(row_comm.iloc[0]["in_flagged_community"])
    community_id = int(row_comm.iloc[0]["community_id"])

    if not in_flagged:
        st.info(
            f"Entity này (community_id={community_id}) không có cặp Khách-Quán nào "
            f"với graph_flag=True trong dữ liệu Bước 3, nên không được đánh dấu là "
            f"thuộc cộng đồng cấu kết."
        )
    else:
        st.success(
            f"✅ Entity thuộc cộng đồng bị flag: **community_id={community_id}** "
            f"— entity này có ít nhất 1 cặp Khách-Quán với graph_flag=True "
            f"(khách hàng/quán, hoặc cùng cộng đồng với 1 khách/quán đã flag nếu là tài xế)."
        )

        # Lấy toàn bộ node cùng community
        comm_nodes_df = df_comm[df_comm["community_id"] == community_id]
        comm_node_ids = comm_nodes_df["node_id"].tolist()

        st.write(f"**Thành viên cộng đồng ({len(comm_node_ids)} node):**")
        st.dataframe(comm_nodes_df[["node_id"]].reset_index(drop=True), width="stretch")

        # Xây dựng subgraph từ graph_edges
        df_edges = load_graph_edges()
        comm_node_set = set(comm_node_ids)
        sub_edges = df_edges[
            df_edges["node_a"].isin(comm_node_set) &
            df_edges["node_b"].isin(comm_node_set)
        ]

        st.write(f"**Cạnh nội bộ cộng đồng ({len(sub_edges)} cạnh):**")
        st.dataframe(sub_edges.reset_index(drop=True), width="stretch")

        # Vẽ subgraph bằng networkx + matplotlib
        G_sub = nx.Graph()
        for _, r in sub_edges.iterrows():
            G_sub.add_edge(r["node_a"], r["node_b"], weight=int(r["weight"]))
        # Thêm node cô lập nếu chưa có
        for n in comm_node_ids:
            if n not in G_sub:
                G_sub.add_node(n)

        # Màu theo loại
        COLOR_MAP = {"C": "#4E9BCD", "D": "#F4A261", "M": "#2A9D8F"}
        node_colors = [COLOR_MAP.get(n.split("_")[0], "#AAAAAA") for n in G_sub.nodes()]

        fig, ax = plt.subplots(figsize=(7, 5))
        pos = nx.spring_layout(G_sub, seed=42, k=1.5)
        nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=800, ax=ax)
        nx.draw_networkx_labels(G_sub, pos, ax=ax, font_size=7)
        edge_labels = {(u, v): d["weight"] for u, v, d in G_sub.edges(data=True)}
        nx.draw_networkx_edges(G_sub, pos, ax=ax, width=1.5, alpha=0.7)
        nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, ax=ax, font_size=8)

        # Legend
        legend_patches = [
            mpatches.Patch(color="#4E9BCD", label="Customer (C_)"),
            mpatches.Patch(color="#F4A261", label="Driver (D_)"),
            mpatches.Patch(color="#2A9D8F", label="Merchant (M_)"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
        ax.set_title(f"Subgraph — Community {community_id}", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ANOMALY
# ══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.subheader("Anomaly Score & Feature Analysis")

    if entity_type in ("customer", "merchant"):
        df_feat = load_cases_with_features()

        # Lấy tất cả dòng liên quan đến entity_id
        col_filter = "customer_id" if entity_type == "customer" else "merchant_id"
        df_entity = df_feat[df_feat[col_filter] == entity_id].copy()

        if len(df_entity) == 0:
            st.info(f"Không tìm thấy `{entity_id}` trong cases_with_features.parquet.")
        else:
            # Lấy dòng có fraud_score cao nhất
            best_row = df_entity.loc[df_entity["fraud_score"].idxmax()]
            cust_id  = best_row["customer_id"]
            merch_id = best_row["merchant_id"]
            fraud_sc = best_row["fraud_score"]
            tier_val = best_row["tier"]

            st.write(
                f"**Hiển thị theo cặp** `{cust_id}` × `{merch_id}` "
                f"— cặp rủi ro cao nhất"
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("fraud_score", f"{fraud_sc:.4f}")
            c2.metric("tier", tier_val)
            c3.metric("graph_flag", str(best_row.get("graph_flag", "N/A")))

            if best_row.get("features_imputed", False):
                st.warning(
                    "⚠️ Dữ liệu feature của cặp này đã được impute (features_imputed=True) "
                    "do số đơn ít — kết quả phân tích feature có thể kém chính xác."
                )

            # 6 features — tính percentile so với toàn bộ phân phối
            FEATURES = [
                ("n_orders",            "Số đơn"),
                ("gmv_mean",            "GMV trung bình"),
                ("cv_gmv",              "CV GMV"),
                ("discount_rate_mean",  "Tỷ lệ discount TB"),
                ("n_promo_unique",      "Số promo khác nhau"),
                ("avg_gap_minutes",     "Gap giữa đơn (phút)"),
            ]

            feat_vals  = []
            feat_labels = []
            feat_pcts  = []

            for feat_col, feat_label in FEATURES:
                if feat_col not in df_feat.columns:
                    continue
                val = best_row.get(feat_col, np.nan)
                if pd.isna(val):
                    pct = 0.0
                else:
                    col_series = df_feat[feat_col].dropna()
                    pct = float((col_series < val).mean()) * 100.0
                feat_labels.append(feat_label)
                feat_vals.append(val)
                feat_pcts.append(pct)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            bars = ax2.barh(feat_labels, feat_pcts, color="#4E9BCD")
            ax2.set_xlim(0, 100)
            ax2.set_xlabel("Percentile so với toàn bộ cặp (%)")
            ax2.set_title("Feature Percentile (so với phân phối toàn bộ cases)")
            for bar, val, pct in zip(bars, feat_vals, feat_pcts):
                ax2.text(
                    min(pct + 1, 95), bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%  (val={val:.2f})" if isinstance(val, float) else f"{pct:.1f}%  (val={val})",
                    va="center", fontsize=8,
                )
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            st.write("**Chi tiết feature:**")
            detail_df = pd.DataFrame({
                "Feature": feat_labels,
                "Giá trị": feat_vals,
                "Percentile (%)": [f"{p:.1f}" for p in feat_pcts],
            })
            st.dataframe(detail_df, width="stretch")

    elif entity_type == "driver":
        df_drv = load_driver_scores()
        row_drv = df_drv[df_drv["driver_id"] == entity_id]

        if len(row_drv) == 0:
            st.info(f"Không tìm thấy driver `{entity_id}` trong driver_scores.parquet.")
        else:
            r = row_drv.iloc[0]
            tier_drv = r["driver_tier"]
            pct_drv  = r["driver_percentile"]

            c1, c2 = st.columns(2)
            c1.metric("driver_tier", tier_drv)
            c2.metric("driver_percentile", f"{pct_drv:.4f}")

            # 4 features driver
            DRV_FEATURES = [
                ("ghost_delivery_ratio",   "Ghost delivery ratio"),
                ("overlap_count",          "Overlap count"),
                ("merchant_concentration", "Merchant concentration"),
                ("targeted_customer_flag", "Targeted customer flag"),
            ]

            feat_labels2 = []
            feat_vals2   = []
            feat_pcts2   = []

            for feat_col, feat_label in DRV_FEATURES:
                if feat_col not in df_drv.columns:
                    continue
                val = r.get(feat_col, np.nan)
                if pd.isna(val):
                    pct = 0.0
                else:
                    col_series = df_drv[feat_col].dropna()
                    pct = float((col_series < val).mean()) * 100.0
                feat_labels2.append(feat_label)
                feat_vals2.append(val)
                feat_pcts2.append(pct)

            fig3, ax3 = plt.subplots(figsize=(7, 3))
            bars3 = ax3.barh(feat_labels2, feat_pcts2, color="#F4A261")
            ax3.set_xlim(0, 100)
            ax3.set_xlabel("Percentile so với toàn bộ driver (%)")
            ax3.set_title("Driver Feature Percentile")
            for bar, val, pct in zip(bars3, feat_vals2, feat_pcts2):
                ax3.text(
                    min(pct + 1, 95), bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%  (val={val})",
                    va="center", fontsize=8,
                )
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            st.write("**Chi tiết feature:**")
            detail_df2 = pd.DataFrame({
                "Feature": feat_labels2,
                "Giá trị": feat_vals2,
                "Percentile (%)": [f"{p:.1f}" for p in feat_pcts2],
            })
            st.dataframe(detail_df2, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: LỊCH SỬ ĐƠN HÀNG
# ══════════════════════════════════════════════════════════════════════════════
with tab_orders:
    st.subheader("Lịch sử đơn hàng")

    df_orders = load_orders()

    # Lọc đơn theo đúng cột của entity_type
    if id_col not in df_orders.columns:
        st.error(f"Cột `{id_col}` không tồn tại trong orders_full.parquet.")
    else:
        df_entity_orders = df_orders[df_orders[id_col] == entity_id].copy()

        # Sort theo order_time_local_tz giảm dần (nếu có)
        time_col = "order_time_local_tz"
        if time_col in df_entity_orders.columns:
            df_entity_orders = df_entity_orders.sort_values(time_col, ascending=False)
        df_entity_orders = df_entity_orders.reset_index(drop=True)

        n_orders = len(df_entity_orders)
        st.write(f"**Tổng số đơn liên quan: {n_orders:,}**")

        if n_orders == 0:
            st.info("Không tìm thấy đơn hàng nào liên quan đến entity này.")
        else:
            st.dataframe(df_entity_orders, width="stretch", height=500)
