"""
app.py
------
Credora Finance | Credit Risk & Loan Approval Analytics Platform
Business Intelligence Dashboard (Streamlit)  --  PRD section 14

All eight required components
  1  Loan approval rate              KPI card + monthly trend line
  2  Default rate                    KPI card + by-loan-type bar
  3  Customer risk distribution      donut: Low / Medium / High
  4  Credit score trends             line (avg bureau score of applicants by month) + histogram
  5  Geographic loan distribution    city map + state bar
  6  Monthly loan applications       column chart
  7  Approval vs rejection trends    stacked columns by month
  8  High-risk customer list         filterable, downloadable table

Filters (sidebar): loan type, state, risk category, application month, score basis.
Reads only datasets/*.csv and models/scored_*.csv, so it runs without the database.
No names, emails or phone numbers are loaded or shown (privacy NFR).

Run:  streamlit run dashboard/app.py      (after:  python src/run_pipeline.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
RISK_COLORS = {"Low Risk": "#2ca02c", "Medium Risk": "#ff7f0e", "High Risk": "#d62728"}
BASIS_LABEL = {
    "disbursed_loan": "Disbursed loan (out-of-fold)",
    "application_not_disbursed": "Application, not disbursed",
    "reference_loan": "No application (reference loan)",
}
CITY_COORDS = {  # city -> (lat, lon)
    "Ahmedabad": (23.0225, 72.5714), "Bengaluru": (12.9716, 77.5946), "Bhopal": (23.2599, 77.4126),
    "Chandigarh": (30.7333, 76.7794), "Chennai": (13.0827, 80.2707), "Coimbatore": (11.0168, 76.9558),
    "Delhi": (28.6139, 77.2090), "Gurugram": (28.4595, 77.0266), "Hyderabad": (17.3850, 78.4867),
    "Indore": (22.7196, 75.8577), "Jaipur": (26.9124, 75.7873), "Kochi": (9.9312, 76.2673),
    "Kolkata": (22.5726, 88.3639), "Lucknow": (26.8467, 80.9462), "Mumbai": (19.0760, 72.8777),
    "Mysuru": (12.2958, 76.6394), "Nagpur": (21.1458, 79.0882), "Noida": (28.5355, 77.3910),
    "Pune": (18.5204, 73.8567), "Surat": (21.1702, 72.8311),
}

st.set_page_config(page_title="Credora Risk & Loan Analytics", layout="wide", page_icon="\U0001F4CA")


@st.cache_data
def load_data():
    customers = pd.read_csv(ROOT / "datasets" / "customers.csv", usecols=["customer_id", "city", "state"])
    credit = pd.read_csv(ROOT / "datasets" / "credit_history.csv", usecols=["customer_id", "credit_score"])
    loans = pd.read_csv(ROOT / "datasets" / "loan_applications.csv", parse_dates=["application_date"])
    loans["month"] = loans["application_date"].dt.to_period("M").astype(str)
    scored_path = ROOT / "models" / "scored_customers.csv"
    scored = pd.read_csv(scored_path) if scored_path.exists() else pd.DataFrame()
    metrics_path = ROOT / "models" / "metrics_report.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    imp_path = ROOT / "models" / "feature_importance.csv"
    imp = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()
    return customers, credit, loans, scored, metrics, imp


customers, credit, loans, scored, metrics, importance = load_data()
if scored.empty:
    st.error("models/scored_customers.csv not found. Run `python src/run_pipeline.py` first.")
    st.stop()

# ---------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
loan_types = sorted(loans["loan_type"].unique())
sel_types = st.sidebar.multiselect("Loan type", loan_types, default=loan_types)
states = sorted(customers["state"].unique())
sel_states = st.sidebar.multiselect("Region / state", states, default=states)
sel_risk = st.sidebar.multiselect("Risk category", RISK_ORDER, default=RISK_ORDER)
months = sorted(loans["month"].unique())
sel_months = st.sidebar.select_slider("Application month", options=months, value=(months[0], months[-1]))
bases = [b for b in BASIS_LABEL if b in set(scored["score_basis"])]
sel_basis = st.sidebar.multiselect("Score basis (customer views)", bases, default=bases,
                                   format_func=lambda b: BASIS_LABEL[b])
st.sidebar.caption(
    "Loan type and month filter application views; state, risk category and score basis filter "
    "customer views. State and risk category apply to both."
)

# ---------------------------------------------------------------- apply filters
cust = customers.merge(scored[["customer_id", "risk_category"]], on="customer_id", how="left")
ok_customers = cust[cust["state"].isin(sel_states) & cust["risk_category"].isin(sel_risk)]["customer_id"]
loans_f = loans[
    loans["loan_type"].isin(sel_types)
    & loans["month"].between(sel_months[0], sel_months[1])
    & loans["customer_id"].isin(ok_customers)
].merge(customers[["customer_id", "city", "state"]], on="customer_id", how="left")
scored_f = scored[scored["state"].isin(sel_states) & scored["risk_category"].isin(sel_risk)
                  & scored["score_basis"].isin(sel_basis)]

# ---------------------------------------------------------------- header + KPI row
st.title("\U0001F4CA Credora Finance \u2014 Credit Risk & Loan Approval Dashboard")
st.caption("Data & Analytics \u2014 Risk Engineering | Decision-support only: the underwriter makes the final call.")

if loans_f.empty and scored_f.empty:
    st.warning("No data matches the current filters. Widen a filter in the sidebar.")
    st.stop()

disbursed = loans_f[loans_f["disbursed_flag"] == 1]
decided = loans_f[loans_f["approval_status"].isin(["Approved", "Rejected"])]
approval_rate = (decided["approval_status"] == "Approved").mean() if len(decided) else float("nan")
default_rate = disbursed["loan_default"].mean() if len(disbursed) else float("nan")
fmt = lambda v: "n/a" if pd.isna(v) else f"{v:.1%}"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Applications", f"{len(loans_f):,}")
k2.metric("Approval rate (decided)", fmt(approval_rate), help="Approved / (Approved + Rejected); excludes Under Review")
k3.metric("Default rate (disbursed)", fmt(default_rate))
k4.metric("Disbursed value", f"\u20b9{loans_f['approved_amount'].sum() / 1e7:,.1f} cr")
k5.metric("High-risk customers", f"{(scored_f['risk_category'] == 'High Risk').sum():,}")
st.divider()

# ---------------------------------------------------------------- 1 & 2
c1, c2 = st.columns(2)
with c1:
    st.subheader("1 \u00b7 Loan approval rate")
    m = decided.groupby("month").agg(applications=("application_id", "count"),
                                     approved=("approval_status", lambda s: (s == "Approved").sum())).reset_index()
    m["approval_rate"] = m["approved"] / m["applications"]
    fig = px.line(m, x="month", y="approval_rate", markers=True)
    fig.update_yaxes(tickformat=".0%", title="Approval rate"); fig.update_xaxes(title="")
    st.plotly_chart(fig, width="stretch")
with c2:
    st.subheader("2 \u00b7 Default rate by loan type")
    d = disbursed.groupby("loan_type")["loan_default"].agg(rate="mean", loans="count").reset_index().sort_values("rate")
    fig = px.bar(d, x="rate", y="loan_type", orientation="h", hover_data=["loans"])
    fig.update_xaxes(tickformat=".0%", title="Default rate"); fig.update_yaxes(title="")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 3 & 4
c3, c4 = st.columns(2)
with c3:
    st.subheader("3 \u00b7 Customer risk distribution")
    dist = scored_f["risk_category"].value_counts().reindex(RISK_ORDER).dropna().reset_index()
    dist.columns = ["risk_category", "customers"]
    fig = px.pie(dist, names="risk_category", values="customers", hole=0.5,
                 color="risk_category", color_discrete_map=RISK_COLORS)
    st.plotly_chart(fig, width="stretch")
    st.caption("Application-based scores are extrapolations (see the Model tab); use the "
               "'Score basis' filter to isolate customers with a disbursed loan.")
with c4:
    st.subheader("4 \u00b7 Credit score trends")
    cs = loans_f.merge(credit, on="customer_id", how="left")
    trend = cs.groupby("month")["credit_score"].mean().reset_index()
    fig = px.line(trend, x="month", y="credit_score", markers=True)
    fig.update_yaxes(title="Avg bureau score of applicants"); fig.update_xaxes(title="")
    st.plotly_chart(fig, width="stretch")
    h = px.histogram(credit[credit["customer_id"].isin(ok_customers)], x="credit_score", nbins=30)
    h.update_layout(height=230, margin=dict(t=10, b=10), yaxis_title="Customers", xaxis_title="Credit score")
    st.plotly_chart(h, width="stretch")
    st.caption("Bureau data is a single snapshot (2024-06-30), so the trend is the average score of "
               "customers applying each month, not a score history.")

# ---------------------------------------------------------------- 5 & 6
c5, c6 = st.columns(2)
with c5:
    st.subheader("5 \u00b7 Geographic loan distribution")
    g = loans_f.groupby("city").agg(applications=("application_id", "count"),
                                    default_rate=("loan_default", "mean"),
                                    disbursed_cr=("approved_amount", lambda s: s.sum() / 1e7)).reset_index()
    g["lat"] = g["city"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    g["lon"] = g["city"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    fig = px.scatter_geo(g.dropna(subset=["lat"]), lat="lat", lon="lon", size="applications",
                         color="default_rate", hover_name="city",
                         hover_data={"lat": False, "lon": False, "disbursed_cr": ":.1f", "default_rate": ":.1%"},
                         color_continuous_scale="RdYlGn_r", size_max=28)
    fig.update_geos(scope="asia", fitbounds="locations", showcountries=True, showland=True)
    fig.update_layout(height=330, margin=dict(t=0, b=0, l=0, r=0), coloraxis_colorbar_title="Default rate")
    st.plotly_chart(fig, width="stretch")
    bs = loans_f.groupby("state").size().sort_values(ascending=False).head(10).reset_index(name="applications")
    st.plotly_chart(px.bar(bs, x="applications", y="state", orientation="h").update_layout(
        height=300, margin=dict(t=10), yaxis={"categoryorder": "total ascending"}), width="stretch")
with c6:
    st.subheader("6 \u00b7 Monthly loan applications")
    mc = loans_f.groupby("month").size().reset_index(name="applications")
    fig = px.bar(mc, x="month", y="applications")
    fig.update_xaxes(title=""); st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 7
st.subheader("7 \u00b7 Approval vs rejection trends")
sm = loans_f.groupby(["month", "approval_status"]).size().reset_index(name="applications")
fig = px.bar(sm, x="month", y="applications", color="approval_status", barmode="stack",
             color_discrete_map={"Approved": "#2ca02c", "Rejected": "#d62728", "Under Review": "#aaaaaa"})
fig.update_xaxes(title=""); st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 8
st.subheader("8 \u00b7 High-risk customer watchlist")
watch = scored_f.sort_values(["risk_score", "default_probability"], ascending=False).copy()
watch["score_basis"] = watch["score_basis"].map(BASIS_LABEL)
st.dataframe(watch[["customer_id", "state", "city", "risk_score", "risk_category", "default_probability",
                    "score_basis", "top_risk_drivers"]].head(500),
             width="stretch", height=380, hide_index=True)
st.caption(f"{len(watch):,} customers match the filters (top 500 shown). "
           "`top_risk_drivers` lists the SHAP features pushing that customer's risk up the most.")
st.download_button("Download filtered watchlist (CSV)",
                   watch.drop(columns=["application_id"]).to_csv(index=False).encode("utf-8"),
                   "high_risk_watchlist.csv", "text/csv")

# ---------------------------------------------------------------- model transparency
st.divider()
st.subheader("Model & explainability")
if metrics:
    t = metrics["champion_test_at_operating_threshold"]
    sc = metrics["success_scorecard"]
    a, b, c, d = st.columns(4)
    a.metric("Champion", metrics["champion_model"])
    b.metric("Test ROC-AUC", f"{sc['roc_auc_achieved']:.3f}", help="PRD target 0.80")
    c.metric("Recall at operating threshold", f"{t['recall']:.3f}", help="PRD target 0.70")
    d.metric("Precision", f"{t['precision']:.3f}")
    bv = pd.DataFrame(metrics["band_validation_test"])
    bv["observed_default_rate"] = bv["observed_default_rate"].map("{:.1%}".format)
    bv["share_of_loans"] = bv["share_of_loans"].map("{:.1%}".format)
    l, r = st.columns(2)
    with l:
        st.markdown("**Do the risk bands rank risk correctly?** (held-out test loans)")
        st.dataframe(bv.rename(columns={"band": "Band", "loans": "Loans", "share_of_loans": "Share",
                                        "observed_default_rate": "Observed default rate"}),
                     hide_index=True, width="stretch")
    with r:
        if not importance.empty:
            top = importance.head(10).sort_values("mean_abs_shap")
            st.plotly_chart(px.bar(top, x="mean_abs_shap", y="feature", orientation="h",
                                   title="Top drivers of default risk (mean |SHAP|)").update_layout(height=320),
                            width="stretch")
