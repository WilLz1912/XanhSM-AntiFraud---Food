"""
pages/5_Dashboard_Tong_Hop.py — Trang 5: Dashboard Tổng hợp
KPI tổng quan, xu hướng case theo ngày, GMV/Discount theo từng rule,
tiến độ xác minh (đọc trực tiếp case_verifications.parquet, tự làm mới).
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Dashboard Tổng hợp — XanhSM Fraud",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(APP_DIR, "data")
RULES_DIR = os.path.join(DATA_DIR, "rules")

CASES_PATH = os.path.join(DATA_DIR, "cases.parquet")
ORDERS_PATH = os.path.join(DATA_DIR, "orders_full.parquet")
DRIVER_SCORES_PATH = os.path.join(DATA_DIR, "driver_scores.parquet")
ENTITY_SUMMARY = {
    "customer": os.path.join(DATA_DIR, "entity_summary_customer.parquet"),
    "merchant": os.path.join(DATA_DIR, "entity_summary_merchant.parquet"),
    "driver":   os.path.join(DATA_DIR, "entity_summary_driver.parquet"),
}
VERIF_PATH = os.path.join(DATA_DIR, "case_verifications.parquet")

# Rule nào dùng cột tiền nào — lấy ĐÚNG cột mỗi rule đã tự tính sẵn,
# không suy ra công thức mới. Xem thảo luận đã chốt với vận hành.
RULE_MONEY_CONFIG = {
    "kb1_trio_collusion.parquet":  {"name": "KB-1 Trio Collusion",          "gmv_col": "gmv_sum",               "discount_col": None,            "note": ""},
    "kb3_merchant_driver.parquet": {"name": "KB-3 Merchant-Driver cố định", "gmv_col": "pair_gmv",              "discount_col": None,            "note": ""},
    "kb4_shell_merchant.parquet":  {"name": "KB-4 Shell Merchant",          "gmv_col": "total_gmv",             "discount_col": "total_discount", "note": ""},
    "kb5_fake_prep_time.parquet":  {"name": "KB-5 Fake Prep Time",          "gmv_col": None,                    "discount_col": "total_discount", "note": ""},
    "kb6_overlap.parquet":         {"name": "KB-Overlap",                   "gmv_col": "food_total_order_amount", "discount_col": "discount",     "note": "GMV đơn liên quan — KHÔNG phải tiền đã mất (rule phát hiện lịch trình bất thường, không trực tiếp gây thất thoát)"},
    "kb10_driver_targeted.parquet": {"name": "KB-10 Driver nhắm hại khách", "gmv_col": None,                    "discount_col": None,            "note": "Không có số tiền — rule dựa trên tỷ lệ huỷ đơn"},
    "kb_rating.parquet":           {"name": "KB-Rating",                    "gmv_col": None,                    "discount_col": None,            "note": "Không có số tiền — rule dựa trên đánh giá (rating)"},
    "split_orders.parquet":        {"name": "Split Orders",                 "gmv_col": "total_gmv",             "discount_col": "total_discount", "note": ""},
    "ghost_delivery.parquet":      {"name": "Ghost Delivery",                "gmv_col": "food_total_order_amount", "discount_col": "discount",   "note": ""},
    "bom_hang.parquet":            {"name": "Bom hàng (COD No-show)",        "gmv_col": "food_total_order_amount", "discount_col": "discount",   "note": "Gần nhất với thất thoát thật — tiền món quán đã chuẩn bị nhưng không thu được"},
}


# ── Data loaders ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_cases():
    return pd.read_parquet(CASES_PATH)


@st.cache_data(ttl=60)
def load_orders():
    return pd.read_parquet(ORDERS_PATH)


@st.cache_data(ttl=60)
def load_driver_scores():
    return pd.read_parquet(DRIVER_SCORES_PATH)


@st.cache_data(ttl=60)
def load_entity_summary(entity_type):
    return pd.read_parquet(ENTITY_SUMMARY[entity_type])


@st.cache_data(ttl=60)
def load_rule_money():
    rows = []
    for fname, cfg in RULE_MONEY_CONFIG.items():
        fpath = os.path.join(RULES_DIR, fname)
        if not os.path.exists(fpath):
            continue
        df_r = pd.read_parquet(fpath)
        gmv_total = float(df_r[cfg["gmv_col"]].sum()) if cfg["gmv_col"] else None
        discount_total = float(df_r[cfg["discount_col"]].sum()) if cfg["discount_col"] else None
        rows.append({
            "rule": cfg["name"], "so_dong": len(df_r),
            "gmv_lien_quan": gmv_total, "discount_lien_quan": discount_total,
            "ghi_chu": cfg["note"],
        })
    return pd.DataFrame(rows)


def load_verifications():
    """Không cache — cần thấy ngay khi Trang 4 vừa lưu."""
    if os.path.exists(VERIF_PATH):
        return pd.read_parquet(VERIF_PATH)
    return pd.DataFrame(columns=[
        "entity_type", "entity_id", "verification_status", "note",
        "score_at_verification", "tier_at_verification", "verified_at",
    ])


@st.cache_data(ttl=60)
def build_daily_trend(_orders, _cases):
    # Ngày cuối cùng trong toàn bộ dữ liệu có thể chưa hết ngày (upload dở dang) —
    # tự phát hiện bằng giờ của bản ghi mới nhất, KHÔNG hard-code ngày cụ thể,
    # để vẫn đúng khi có dữ liệu mới qua Trang 1 sau này. Ngưỡng 23:00 — nếu đơn
    # cuối cùng trong ngày đó đến trước 23:00 thì coi là ngày chưa đầy đủ, loại
    # khỏi đường trend (không vẽ như 1 điểm dữ liệu thật, tránh gây hiểu nhầm
    # là số liệu sụt giảm).
    max_ts = _orders["order_time_local_tz"].max()
    partial_info = None
    if pd.notna(max_ts) and max_ts.time() < pd.Timestamp("23:00").time():
        partial_info = {"date": max_ts.date(), "cutoff": max_ts.strftime("%H:%M")}

    flagged = _cases[_cases.tier != "Normal"][["customer_id", "merchant_id", "tier"]]
    o2 = _orders.merge(flagged, on=["customer_id", "merchant_id"], how="inner")
    o2["ngay"] = pd.to_datetime(o2["order_time_local_tz"]).dt.date

    if partial_info is not None:
        o2 = o2[o2["ngay"] != partial_info["date"]]

    daily = o2.groupby(["ngay", "tier"]).apply(
        lambda g: g[["customer_id", "merchant_id"]].drop_duplicates().shape[0],
        include_groups=False,
    ).unstack(fill_value=0)
    for col in ["High", "Medium"]:
        if col not in daily.columns:
            daily[col] = 0
    daily["Tổng"] = daily["High"] + daily["Medium"]
    daily = daily.sort_index().reset_index()

    # Chèn điểm trống (NaN) cho khoảng thiếu dữ liệu >1 ngày để KHÔNG vẽ
    # nhầm thành đường liền mạch giảm dần qua khoảng trống thật.
    full_rows = []
    for i in range(len(daily)):
        full_rows.append(daily.iloc[i])
        if i < len(daily) - 1:
            gap = (daily.iloc[i + 1]["ngay"] - daily.iloc[i]["ngay"]).days
            if gap > 1:
                mid = pd.Timestamp(daily.iloc[i]["ngay"]) + pd.Timedelta(days=gap / 2)
                full_rows.append(pd.Series({
                    "ngay": mid.date(), "High": None, "Medium": None, "Tổng": None,
                }))
    return pd.DataFrame(full_rows).reset_index(drop=True), partial_info


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st_autorefresh(interval=15_000, key="dashboard_autorefresh")

st.title("Dashboard Tổng hợp")
st.caption(
    "Tự làm mới mỗi 15 giây — trạng thái xác minh (Trang 4) sẽ cập nhật gần như ngay lập tức. "
    "Các số liệu khác cache 60 giây."
)

df_cases = load_cases()
df_orders = load_orders()
df_driver = load_driver_scores()
df_verif = load_verifications()

# ── Hàng KPI 1: quy mô hệ thống ──────────────────────────────────────────────
st.subheader("Tổng quan hệ thống")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Tổng đơn hàng", f"{len(df_orders):,}")
k2.metric("Case Khách-Quán High", f"{int((df_cases.tier=='High').sum()):,}")
k3.metric("Case Khách-Quán Medium", f"{int((df_cases.tier=='Medium').sum()):,}")
k4.metric("Driver High", f"{int((df_driver.driver_tier=='High').sum()):,}")
k5.metric("Driver Medium", f"{int((df_driver.driver_tier=='Medium').sum()):,}")

st.divider()

# ── Hàng KPI 2: xác minh (real-time) ─────────────────────────────────────────
st.subheader("Tiến độ Xác minh (real-time)")
n_verif = len(df_verif)
n_co = int((df_verif.verification_status == "Có").sum()) if n_verif else 0
n_khong = int((df_verif.verification_status == "Không").sum()) if n_verif else 0
n_theodoi = int((df_verif.verification_status == "Theo dõi").sum()) if n_verif else 0

n_queue_total = int((df_cases.tier != "Normal").customer_id.count()) if False else None
# Tổng hàng chờ = số entity High/Medium duy nhất trên cả 3 loại
n_queue_total = sum(
    int((load_entity_summary(et)[
        "tier_worst" if et != "driver" else "driver_tier"
    ] != "Normal").sum())
    for et in ["customer", "driver", "merchant"]
)
n_pending = n_queue_total - n_verif

v1, v2, v3, v4, v5 = st.columns(5)
v1.metric("Chờ xác minh", f"{max(n_pending,0):,}")
v2.metric("✅ Confirm đúng (Có)", f"{n_co:,}")
v3.metric("❌ Confirm sai (Không)", f"{n_khong:,}")
v4.metric("👀 Theo dõi", f"{n_theodoi:,}")
precision = (n_co / (n_co + n_khong) * 100) if (n_co + n_khong) > 0 else None
v5.metric("Precision xác minh", f"{precision:.1f}%" if precision is not None else "—")
if n_verif == 0:
    st.info("Chưa có case nào được xác minh trên Trang 4.")

st.divider()

# ── Xu hướng case theo ngày ───────────────────────────────────────────────────
st.subheader("Xu hướng Case theo Ngày")
st.caption(
    "Mỗi điểm = số case Khách-Quán (High/Medium) DUY NHẤT có ít nhất 1 đơn phát sinh trong ngày đó "
    "(không phải số đơn). Đường đứt đoạn = không có dữ liệu (không phải 0 case)."
)

trend, partial_info = build_daily_trend(df_orders, df_cases)
avg_total = trend["Tổng"].dropna().mean()

fig = go.Figure()
for col, color in [("Tổng", "#4E9BCD"), ("High", "#E76F51"), ("Medium", "#F4A261")]:
    fig.add_trace(go.Scatter(
        x=trend["ngay"], y=trend[col], mode="lines+markers", name=col,
        line=dict(color=color, width=2), connectgaps=False,
    ))
fig.update_layout(
    xaxis_title="Ngày", yaxis_title="Số case duy nhất",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=40, b=20), height=420,
)
st.plotly_chart(fig, width="stretch")

n_days_real = trend["Tổng"].notna().sum()
partial_note = ""
if partial_info is not None:
    partial_note = (
        f" ⚠️ Ngày {partial_info['date']} có dữ liệu nhưng chỉ đến ~{partial_info['cutoff']} "
        f"(chưa hết ngày) nên đã LOẠI KHỎI biểu đồ — không so sánh được với các ngày đủ 24h."
    )
st.caption(
    f"Trung bình **{avg_total:.0f} case/ngày** (High+Medium, tính trên {n_days_real} ngày đủ dữ liệu)."
    f"{partial_note} Đường đứt đoạn trong biểu đồ = không có dữ liệu, không phải 0 case."
)

st.divider()

# ── GMV / Discount theo từng rule ─────────────────────────────────────────────
st.subheader("Quy mô Tiền theo từng Rule")
st.caption(
    "GMV/Discount lấy TRỰC TIẾP từ số liệu mỗi rule đã tự tính sẵn — không suy ra công thức mới. "
    "KHÔNG cộng tổng across rule vì có đơn trùng giữa 1 số rule (vd Ghost Delivery ∩ KB-Overlap)."
)

df_rule_money = load_rule_money()
df_chart = df_rule_money.dropna(subset=["gmv_lien_quan"]).sort_values("gmv_lien_quan", ascending=True)

fig2 = px.bar(
    df_chart, x="gmv_lien_quan", y="rule", orientation="h",
    labels={"gmv_lien_quan": "GMV liên quan (VNĐ)", "rule": ""},
    color="gmv_lien_quan", color_continuous_scale="Oranges",
)
fig2.update_layout(height=380, margin=dict(t=20, b=20), coloraxis_showscale=False)
st.plotly_chart(fig2, width="stretch")

display_money = df_rule_money.copy()
display_money["gmv_lien_quan"] = display_money["gmv_lien_quan"].apply(
    lambda x: f"{x:,.0f} đ" if pd.notna(x) else "N/A"
)
display_money["discount_lien_quan"] = display_money["discount_lien_quan"].apply(
    lambda x: f"{x:,.0f} đ" if pd.notna(x) else "N/A"
)
st.dataframe(
    display_money.rename(columns={
        "rule": "Rule", "so_dong": "Số dòng (đơn/case)",
        "gmv_lien_quan": "GMV liên quan", "discount_lien_quan": "Discount liên quan",
        "ghi_chu": "Ghi chú",
    }),
    width="stretch", height=390,
)

st.divider()

# ── Phân bố xác minh theo entity_type ────────────────────────────────────────
st.subheader("Phân bố Xác minh theo Loại Entity")
if len(df_verif):
    dist = df_verif.groupby(["entity_type", "verification_status"]).size().reset_index(name="count")
    fig3 = px.bar(
        dist, x="entity_type", y="count", color="verification_status",
        barmode="group",
        labels={"entity_type": "Loại entity", "count": "Số lượng", "verification_status": "Trạng thái"},
        color_discrete_map={"Có": "#2A9D8F", "Không": "#E76F51", "Theo dõi": "#F4A261"},
    )
    fig3.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig3, width="stretch")
else:
    st.info("Chưa có dữ liệu xác minh để vẽ biểu đồ phân bố.")
