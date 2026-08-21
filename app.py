"""
app.py — XanhSM Food Fraud Detection Pipeline (Trang 1: Nạp, Chạy Rule & Chạy Model)
Bao gồm:
  - BƯỚC 1: Upload & Dedupe Dữ liệu Orders
  - BƯỚC 2: Chạy 10 Rule-based & Tạo Chỉ mục Thực thể (Rule Entity Index)
  - BƯỚC 3: Chạy Isolation Forest (Pair + Driver) & Louvain Community Detection
Chỉ sử dụng component mặc định của Streamlit.
"""

import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from sklearn.ensemble import IsolationForest
import streamlit as st

st.set_page_config(
    page_title="XanhSM Fraud Pipeline — Data, Rules & Models",
    layout="wide",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
RULES_DIR = os.path.join(DATA_DIR, "rules")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RULES_DIR, exist_ok=True)

ORDERS_FULL_PATH = os.path.join(DATA_DIR, "orders_full.parquet")
ENTITY_INDEX_PATH = os.path.join(DATA_DIR, "rule_entity_index.parquet")
CASES_PATH = os.path.join(DATA_DIR, "cases.parquet")
FEATURES_PATH = os.path.join(DATA_DIR, "cases_with_features.parquet")
DRIVER_SCORES_PATH = os.path.join(DATA_DIR, "driver_scores.parquet")
GRAPH_EDGES_PATH = os.path.join(DATA_DIR, "graph_edges.parquet")
GRAPH_COMM_PATH = os.path.join(DATA_DIR, "graph_communities.parquet")

ROOT_DIR = os.path.dirname(APP_DIR)
BASELINE_F1 = os.path.join(ROOT_DIR, "orders_food_masked_2026-07-12_2026-07-14.parquet")
BASELINE_F2 = os.path.join(ROOT_DIR, "orders_food_masked_2026-07-24_2026-07-31.parquet")


def load_baseline_orders() -> pd.DataFrame:
    df1 = pd.read_parquet(BASELINE_F1)
    df2 = pd.read_parquet(BASELINE_F2)
    df_merged = pd.concat([df1, df2], ignore_index=True)
    df_merged.columns = [c.strip() for c in df_merged.columns]
    return df_merged


def get_current_orders() -> pd.DataFrame:
    if os.path.exists(ORDERS_FULL_PATH):
        return pd.read_parquet(ORDERS_FULL_PATH)
    else:
        df_base = load_baseline_orders()
        df_base.to_parquet(ORDERS_FULL_PATH, index=False)
        return df_base


# ── Title & Intro ──────────────────────────────────────────────────────────────
st.title("XanhSM Food Fraud Detection — Data & Model Pipeline")
st.caption("Trang 1: Nạp dữ liệu, Khử trùng lặp, Chạy 10 Rule-based và Mô hình Học máy (Isolation Forest + Louvain)")

df_current = get_current_orders()

# ═══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: TRẠNG THÁI & UPLOAD DEDUPE
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("1. Dữ liệu Orders & Khử trùng lặp (Dedupe)")
m1, m2, m3 = st.columns(3)
m1.metric("Tổng số dòng đơn hàng", f"{len(df_current):,}")
m2.metric("Số order_id duy nhất", f"{df_current['order_id'].nunique():,}")
m3.metric("Số lượng cột", f"{len(df_current.columns)}")

uploaded_file = st.file_uploader("Upload file orders mới (.parquet hoặc .csv)", type=["parquet", "csv"])

if uploaded_file is not None:
    st.write(f"File tải lên: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.2f} MB)")
    if st.button("Tiến hành Gộp & Dedupe", type="primary"):
        with st.spinner("Đang đọc và khử trùng lặp..."):
            if uploaded_file.name.endswith(".csv"):
                df_new = pd.read_csv(uploaded_file)
            else:
                df_new = pd.read_parquet(uploaded_file)
            df_new.columns = [c.strip() for c in df_new.columns]
            n_new = len(df_new)
            df_combined = pd.concat([df_current, df_new], ignore_index=True)
            n_before = len(df_combined)
            df_deduped = df_combined.drop_duplicates(subset="order_id", keep="first").reset_index(drop=True)
            n_after = len(df_deduped)
            n_duplicates = n_before - n_after
            df_deduped.to_parquet(ORDERS_FULL_PATH, index=False)

            st.success("Khử trùng lặp thành công!")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Số dòng file mới", f"{n_new:,}")
            r2.metric("Số dòng trước dedupe", f"{n_before:,}")
            r3.metric("Dòng trùng bị loại", f"{n_duplicates:,}")
            r4.metric("Số dòng sau dedupe", f"{n_after:,}")
            st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: CHẠY 10 RULES & TẠO CHỈ MỤC THỰC THỂ
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("2. Chạy 10 Rule-based Logic & Bảng Chỉ mục Thực thể")

rule_files = [
    ("KB-1 Trio Collusion", "kb1_trio_collusion.parquet", "customer_id", "driver_id", "merchant_id"),
    ("KB-3 Merchant-Driver cố định", "kb3_merchant_driver.parquet", None, "driver_id", "merchant_id"),
    ("KB-4 Shell Merchant", "kb4_shell_merchant.parquet", "customer_id", None, "merchant_id"),
    ("KB-5 Fake prep time", "kb5_fake_prep_time.parquet", None, None, "merchant_id"),
    ("KB-Overlap", "kb6_overlap.parquet", None, "driver_id", None),
    ("KB-10 Driver nhắm hại khách", "kb10_driver_targeted.parquet", "customer_id", "driver_id", None),
    ("KB-Rating", "kb_rating.parquet", "customer_id", None, None),
    ("Split Orders", "split_orders.parquet", "customer_id", None, "merchant_id"),
    ("Ghost Delivery", "ghost_delivery.parquet", "customer_id", "driver_id", "merchant_id"),
    ("Bom hàng (COD No-show)", "bom_hang.parquet", "customer_id", "driver_id", "merchant_id"),
]

saved_summary = []
for rname, fname, ccol, dcol, mcol in rule_files:
    fpath = os.path.join(RULES_DIR, fname)
    if os.path.exists(fpath):
        df_r = pd.read_parquet(fpath)
        saved_summary.append({
            'rule_name': rname,
            'total_cases': len(df_r),
            'unique_customers': df_r[ccol].nunique() if ccol and ccol in df_r.columns else 0,
            'unique_drivers': df_r[dcol].nunique() if dcol and dcol in df_r.columns else (df_r[df_r['entity_type']=='driver']['entity_id'].nunique() if rname=='KB-Rating' else 0),
            'unique_merchants': df_r[mcol].nunique() if mcol and mcol in df_r.columns else (df_r[df_r['entity_type']=='merchant']['entity_id'].nunique() if rname=='KB-Rating' else 0),
        })

if saved_summary:
    st.write("**Bảng Tổng kết 10 Rule hiện có:**")
    st.dataframe(pd.DataFrame(saved_summary), width='stretch')
else:
    st.info("Chưa có dữ liệu 10 rule. Hãy bấm nút bên dưới để tính toán.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: ISOLATION FOREST & LOUVAIN COMMUNITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("3. Mô hình Học máy (Isolation Forest + Louvain)")
st.write("Tính điểm rủi ro gian lận Fraud Score (cho Cặp Khách - Quán) và Driver Fraud Score (cho Tài xế).")

if os.path.exists(CASES_PATH) and os.path.exists(DRIVER_SCORES_PATH):
    df_cases_view = pd.read_parquet(CASES_PATH)
    df_driver_view = pd.read_parquet(DRIVER_SCORES_PATH)

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.write(f"**Phân phối Tier Cặp Khách - Quán ({len(df_cases_view):,} cặp):**")
        st.dataframe(df_cases_view["tier"].value_counts().reset_index(), width='stretch')
    with c_col2:
        st.write(f"**Phân phối Tier Tài xế ({len(df_driver_view):,} tài xế):**")
        st.dataframe(df_driver_view["driver_tier"].value_counts().reset_index(), width='stretch')

    st.write("**Kiểm tra 2 Case Demo:**")
    demo_check = df_cases_view[df_cases_view["customer_id"].isin(["e1bccf5c09705f63", "52b1fa5ae9febf0c"])]
    st.dataframe(demo_check, width='stretch')

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3 - PHẦN B: DỰNG LẠI DỮ LIỆU CHI TIẾT ĐỒ THỊ CHO TAB GRAPH (TRANG 3)
# ═══════════════════════════════════════════════════════════════════════════════
# QUAN TRỌNG: cases.parquet (fraud_score, tier, graph_flag) là DỮ LIỆU GỐC ĐÃ CHỐT,
# KHÔNG được tính lại ở đây — code Isolation Forest + Louvain gốc sinh ra nó đã bị
# mất (không có trong git, không còn bản sao). Phần dưới đây CHỈ dựng lại 2 file
# chi tiết đồ thị (graph_edges, graph_communities) để trang 3 vẽ subgraph — dùng
# cases.parquet làm chân lý duy nhất, không suy đoán ngưỡng Louvain gốc.
#
# Định nghĩa in_flagged_community (entity-level, khớp cách trang 3 hiển thị theo
# từng entity_id, không theo từng cặp):
#   - node C_x = True  <=>  cases.parquet có dòng customer_id=x và graph_flag=True
#   - node M_y = True  <=>  cases.parquet có dòng merchant_id=y và graph_flag=True
#   - node D_z = True  <=>  D_z cùng community Louvain với ít nhất 1 node C/M đã flag
#     (chỉ để hiển thị đúng tài xế liên quan trong subgraph, vd Case 2 demo)
st.subheader("3B. Dựng lại Đồ thị Cộng đồng (Louvain) cho Tab Graph — Trang 3")
st.caption(
    "cases.parquet KHÔNG bị tính lại ở bước này. Chỉ dựng graph_edges.parquet và "
    "graph_communities.parquet để trang 3 vẽ subgraph, dựa trên graph_flag đã có "
    "sẵn trong cases.parquet."
)

if os.path.exists(GRAPH_EDGES_PATH) and os.path.exists(GRAPH_COMM_PATH):
    df_ge = pd.read_parquet(GRAPH_EDGES_PATH)
    df_gc = pd.read_parquet(GRAPH_COMM_PATH)
    g1, g2, g3 = st.columns(3)
    g1.metric("Số cạnh (weight ≥ 5)", f"{len(df_ge):,}")
    g2.metric("Số node trong đồ thị", f"{len(df_gc):,}")
    g3.metric("Node thuộc cộng đồng bị flag", f"{int(df_gc['in_flagged_community'].sum()):,}")
else:
    st.info("Chưa có graph_edges.parquet / graph_communities.parquet.")

if st.button("Dựng lại Graph Community (Louvain) từ cases.parquet"):
    if not os.path.exists(CASES_PATH):
        st.error("Chưa có cases.parquet — không thể dựng đồ thị.")
    else:
        with st.spinner("Đang dựng đồ thị C-D-M và chạy Louvain..."):
            df_orders = pd.read_parquet(ORDERS_FULL_PATH)

            edges_cd = (
                df_orders.assign(u="C_" + df_orders["customer_id"], v="D_" + df_orders["driver_id"])
                .groupby(["u", "v"], sort=False).size().rename("weight").reset_index()
            )
            edges_dm = (
                df_orders.assign(u="D_" + df_orders["driver_id"], v="M_" + df_orders["merchant_id"])
                .groupby(["u", "v"], sort=False).size().rename("weight").reset_index()
            )
            edges_cm = (
                df_orders.assign(u="C_" + df_orders["customer_id"], v="M_" + df_orders["merchant_id"])
                .groupby(["u", "v"], sort=False).size().rename("weight").reset_index()
            )
            all_edges = pd.concat([edges_cd, edges_dm, edges_cm], ignore_index=True)
            all_edges = all_edges.groupby(["u", "v"], sort=False)["weight"].sum().reset_index()

            edges_f = all_edges[all_edges["weight"] >= 5]
            G = nx.Graph()
            for _, row in edges_f.iterrows():
                G.add_edge(row["u"], row["v"], weight=int(row["weight"]))

            partition = community_louvain.best_partition(G, weight="weight", random_state=42)
            comm_to_nodes = {}
            for node, cid in partition.items():
                comm_to_nodes.setdefault(cid, set()).add(node)

            comm_density = {}
            for cid, nodes in comm_to_nodes.items():
                sz = len(nodes)
                mx = sz * (sz - 1) // 2
                comm_density[cid] = (G.subgraph(nodes).number_of_edges() / mx) if mx else 0.0

            df_cases_full = pd.read_parquet(CASES_PATH)
            flagged_customers = set(df_cases_full.loc[df_cases_full.graph_flag == True, "customer_id"])
            flagged_merchants = set(df_cases_full.loc[df_cases_full.graph_flag == True, "merchant_id"])
            flagged_node_set = {"C_" + c for c in flagged_customers} | {"M_" + m for m in flagged_merchants}

            comm_rows = []
            for n, cid in partition.items():
                if n[0] in ("C", "M"):
                    flag = n in flagged_node_set
                else:
                    flag = any((x in flagged_node_set) for x in comm_to_nodes[cid])
                comm_rows.append({
                    "node_id": n, "community_id": cid,
                    "density": comm_density[cid], "in_flagged_community": flag,
                })
            df_comm_out = pd.DataFrame(comm_rows)

            out_e = edges_f.rename(columns={"u": "node_a", "v": "node_b"})[
                ["node_a", "node_b", "weight"]].reset_index(drop=True)
            out_e["weight"] = out_e["weight"].astype(int)
            out_e.to_parquet(GRAPH_EDGES_PATH, index=False)
            df_comm_out.to_parquet(GRAPH_COMM_PATH, index=False)

            st.success(
                f"Đã dựng lại: {len(out_e):,} cạnh, {len(df_comm_out):,} node, "
                f"{int(df_comm_out['in_flagged_community'].sum()):,} node bị flag."
            )
            st.rerun()

st.divider()

if st.button("Khôi phục dữ liệu 10 ngày ban đầu (Reset All Data)"):
    with st.spinner("Đang khôi phục..."):
        df_base = load_baseline_orders()
        df_base.to_parquet(ORDERS_FULL_PATH, index=False)
        st.success("Đã khôi phục về dữ liệu 10 ngày chuẩn!")
        st.rerun()
