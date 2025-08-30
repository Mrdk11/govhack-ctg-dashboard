
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO

# Optional PDF (reportlab)
try:
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

st.set_page_config(page_title="Closing the Gap - Multiplier Map", layout="wide")

DATA_HINT = """
Place these files alongside app.py (same folder) or in a ./data subfolder:

Required for core pages:
- ctg-202507-ctg01-healthy-lives-dataset.csv
- ctg-202507-ctg04-early-years-dataset.csv
- ctg-202507-ctg10-justice-dataset.csv
- ctg-202507-ctg12-child-protection-dataset.csv
- ctg-202507-ctg14-wellbeing-dataset.csv
- ctg-202507-ctg17-digital-inclusion-dataset.csv

Optional (enhanced features):
- ctg-202507-ctg02-healthy-birthweight-dataset.csv   # enables T2 positive deviance
- priority_reforms_proxy_by_state.csv                 # PR proxy bar chart & Playbooks metric
- priority_reforms_signals.csv                        # deeper PR indicators (state, signal, weight)
- step4_t4_top10.csv                                  # saved positive deviance output (AEDC)
"""

st.sidebar.title("CtG - Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Regional drilldown", "Playbooks", "Map view", "About"])

def find_file(name: str):
    p = Path(name)
    if p.exists():
        return p
    p2 = Path("data") / name
    return p2 if p2.exists() else None

# ---- Load core datasets (Targets 1,4,10,12,14,17) ----
RAW_FILES = {
    "T1": "ctg-202507-ctg01-healthy-lives-dataset.csv",
    "T2": "ctg-202507-ctg02-healthy-birthweight-dataset.csv",  # optional
    "T4": "ctg-202507-ctg04-early-years-dataset.csv",
    "T10": "ctg-202507-ctg10-justice-dataset.csv",
    "T12": "ctg-202507-ctg12-child-protection-dataset.csv",
    "T14": "ctg-202507-ctg14-wellbeing-dataset.csv",
    "T17": "ctg-202507-ctg17-digital-inclusion-dataset.csv",
}

def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="latin1")

loaded = {}
missing = []
for k, fname in RAW_FILES.items():
    fp = find_file(fname)
    if fp is None and k != "T2":  # T2 is optional
        missing.append(fname)
    elif fp is not None:
        loaded[k] = load_csv(fp)

if missing:
    st.warning("Missing files:\n" + "\n".join(f"- {m}" for m in missing))
    st.info(DATA_HINT)

# Optional proxies
pr_proxy = None
pr_proxy_path = find_file("priority_reforms_proxy_by_state.csv")
if pr_proxy_path:
    pr_proxy = pd.read_csv(pr_proxy_path)

pr_signals = None
pr_signals_path = find_file("priority_reforms_signals.csv")
if pr_signals_path:
    pr_signals = pd.read_csv(pr_signals_path)
    # Compute a 0-100 index from signals, similar to proxy
    agg = pr_signals.groupby("state")["weight"].sum().reset_index(name="points")
    maxp = agg["points"].max() if not agg.empty else 1
    agg["proxy_index_0_100"] = (agg["points"] / maxp * 100).round().astype(int)
    # Merge with existing proxy if present, prefer signals index when available
    if pr_proxy is not None:
        pr_proxy = pr_proxy.drop(columns=["proxy_index_0_100","points"], errors="ignore").merge(
            agg, on="state", how="outer"
        )
    else:
        pr_proxy = agg.copy()

STATE_COLS = ["NSW","Vic","Qld","WA","SA","Tas","ACT","NT","Aust"]
ALL_STATES = ["NSW","Vic","Qld","WA","SA","Tas","ACT","NT"]

TARGET_CONFIG = {
    "T1": {"label": "Healthy lives (incl. life expectancy)", "measure_hint": "life expectancy", "lower_is_better": False},
    "T2": {"label": "Healthy birthweight", "measure_hint": "birthweight|healthy birth", "lower_is_better": False},
    "T4": {"label": "Early years (AEDC - all 5 domains)", "measure_hint": "developmentally on track", "lower_is_better": False},
    "T10":{"label": "Justice - adult imprisonment", "measure_hint": "imprisonment rate", "lower_is_better": True},
    "T12":{"label": "Children in out-of-home care", "measure_hint": "out-of-home care|out of home care", "lower_is_better": True},
    "T14":{"label": "Social & emotional wellbeing (suicide proxy)", "measure_hint": "suicide|psychological", "lower_is_better": True},
    "T17":{"label": "Digital inclusion / access", "measure_hint": "internet|online|digital|broadband|device", "lower_is_better": False},
}

def latest_actual(df: pd.DataFrame, filter_measure_substr=None, ind_only=True, sex="All people"):
    x = df.copy()
    if filter_measure_substr:
        x = x[x["Measure"].str.contains(filter_measure_substr, case=False, na=False)]
    if ind_only:
        x = x[x["Indigenous_Status"].str.contains("Aboriginal and Torres Strait Islander people", na=False)]
    if sex:
        x = x[x["Sex"] == sex]
    if "Description3" in x.columns:
        x = x[x["Description3"] == "Actual"]
    x["Year_num"] = pd.to_numeric(x["Year"], errors="coerce")
    y = x.dropna(subset=["Year_num"]).sort_values("Year_num", ascending=False)
    if y.empty:
        return None
    latest_year = int(y["Year_num"].iloc[0])
    y = y[y["Year_num"] == latest_year]
    return latest_year, y

def latest_trajectory(df: pd.DataFrame, filter_measure_substr=None, ind_only=True, sex="All people"):
    x = df.copy()
    if filter_measure_substr:
        x = x[x["Measure"].str.contains(filter_measure_substr, case=False, na=False)]
    if ind_only:
        x = x[x["Indigenous_Status"].str.contains("Aboriginal and Torres Strait Islander people", na=False)]
    if sex:
        x = x[x["Sex"] == sex]
    if "Description3" in x.columns:
        x = x[x["Description3"] == "Trajectory"]
    x["Year_num"] = pd.to_numeric(x["Year"], errors="coerce")
    y = x.dropna(subset=["Year_num"]).sort_values("Year_num", ascending=False)
    if y.empty:
        return None
    latest_year = int(y["Year_num"].iloc[0])
    y = y[y["Year_num"] == latest_year]
    return latest_year, y

def status_from_actual_vs_traj(actual_val, traj_val, lower_is_better=False):
    if pd.isna(actual_val) or pd.isna(traj_val):
        return "Unknown"
    if lower_is_better:
        if actual_val <= traj_val:
            return "On track"
        elif actual_val <= traj_val * 1.1:
            return "Not on track"
        else:
            return "Worsening"
    else:
        if actual_val >= traj_val:
            return "On track"
        elif actual_val >= traj_val * 0.9:
            return "Not on track"
        else:
            return "Worsening"

def national_status_table() -> pd.DataFrame:
    rows = []
    for key, cfg in TARGET_CONFIG.items():
        if key not in loaded:
            continue
        if key == "T2" and "T2" not in loaded:
            continue
        df = loaded[key]
        la = latest_actual(df, filter_measure_substr=cfg["measure_hint"])
        lt = latest_trajectory(df, filter_measure_substr=cfg["measure_hint"])
        if not la or not lt:
            rows.append({"Target": key, "Name": cfg["label"], "Year": None, "National": None, "Status": "Unknown"})
            continue
        y_a, da = la
        y_t, dt = lt
        nat_a = pd.to_numeric(da["Aust"], errors="coerce").dropna()
        nat_t = pd.to_numeric(dt["Aust"], errors="coerce").dropna()
        a = nat_a.iloc[0] if not nat_a.empty else np.nan
        t = nat_t.iloc[0] if not nat_t.empty else np.nan
        status = status_from_actual_vs_traj(a, t, lower_is_better=cfg["lower_is_better"])
        rows.append({"Target": key, "Name": cfg["label"], "Year": y_a, "National": a, "Trajectory": t, "Status": status})
    return pd.DataFrame(rows)

def state_series(df: pd.DataFrame, cfg: dict, state_code: str) -> pd.DataFrame:
    x = df.copy()
    x = x[x["Measure"].str.contains(cfg["measure_hint"], case=False, na=False)]
    x = x[x["Indigenous_Status"].str.contains("Aboriginal and Torres Strait Islander people", na=False)]
    x = x[x["Sex"] == "All people"]
    x["Year_num"] = pd.to_numeric(x["Year"], errors="coerce")
    xa = x[x["Description3"] == "Actual"].dropna(subset=["Year_num"]).sort_values("Year_num")
    df_state = xa[["Year_num", state_code]].rename(columns={state_code:"state"})
    df_nat = xa[["Year_num", "Aust"]].rename(columns={"Aust":"national"})
    ts = df_state.merge(df_nat, on="Year_num", how="outer")
    xt = x[x["Description3"] == "Trajectory"][["Year_num", state_code, "Aust"]].rename(columns={state_code:"traj_state", "Aust":"traj_nat"})
    ts = ts.merge(xt, on="Year_num", how="outer")
    return ts

def positive_deviance_latest(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    la = latest_actual(df, cfg["measure_hint"])
    if not la:
        return pd.DataFrame()
    y, da = la
    # melt states
    states = da[["Year"] + [c for c in STATE_COLS if c in da.columns]].copy()
    melted = states.melt(id_vars=["Year"], value_vars=[c for c in ALL_STATES if c in states.columns],
                         var_name="state", value_name="value")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["value"])
    mean = melted["value"].mean()
    std = melted["value"].std(ddof=0) if melted["value"].std(ddof=0) != 0 else np.nan
    melted["z_score"] = (melted["value"] - mean) / std if not np.isnan(std) else np.nan
    melted["year"] = y
    return melted.sort_values("z_score", ascending=False)

# Simple state centroid coords for a "map-lite" bubble plot
STATE_CENTROIDS = {
    "NSW": (-31.0, 147.0),
    "Vic": (-36.5, 144.0),
    "Qld": (-20.0, 143.0),
    "WA": (-26.0, 121.0),
    "SA": (-30.0, 135.0),
    "Tas": (-42.0, 147.0),
    "ACT": (-35.5, 149.0),
    "NT": (-19.0, 133.0),
}

def map_dataframe_from_pd(pd_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in pd_df.iterrows():
        st_code = r["state"]
        if st_code in STATE_CENTROIDS:
            lat, lon = STATE_CENTROIDS[st_code]
            rows.append({"lat": lat, "lon": lon, "state": st_code, "value": r["value"], "z_score": r["z_score"]})
    return pd.DataFrame(rows)

def render_traffic_cell(status: str) -> str:
    color = {"On track":"#1f8a70", "Not on track":"#e3a008", "Worsening":"#d64545", "Unknown":"#6b7280"}.get(status, "#6b7280")
    return f"<div style='background:{color};color:white;padding:4px 8px;border-radius:6px;text-align:center'>{status}</div>"

# ---------------- PAGES -----------------
if page == "Overview":
    st.markdown("## National Overview")
    st.caption("Traffic-light summary uses latest Actual vs Trajectory for each target (Indigenous, All people, national).")
    if loaded:
        table = national_status_table()
        if not table.empty:
            table_disp = table.copy()
            table_disp["Status"] = table_disp["Status"].apply(render_traffic_cell)
            st.write("### Status across Targets")
            st.write(table_disp.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No data loaded yet.")
    else:
        st.info("Upload datasets to see the overview.")

    st.write("---")
    st.write("### Priority Reforms Proxy (by state)")
    if pr_proxy is not None and "proxy_index_0_100" in pr_proxy.columns:
        st.bar_chart(pr_proxy.set_index("state")["proxy_index_0_100"])
        st.dataframe(pr_proxy)
    elif pr_proxy is not None:
        st.dataframe(pr_proxy)
        st.caption("Provide 'proxy_index_0_100' or 'signals' with weights to see bar chart.")
    else:
        st.caption("Optional: add `priority_reforms_proxy_by_state.csv` or `priority_reforms_signals.csv`.")

elif page == "Regional drilldown":
    st.markdown("## Regional Drilldown")
    if not loaded:
        st.info("Upload the CtG CSVs to enable drilldown.")
    else:
        target = st.selectbox("Select target", [k for k in TARGET_CONFIG.keys() if k in loaded], format_func=lambda k: f"{k} - {TARGET_CONFIG[k]['label']}")
        state = st.selectbox("Select state/territory", ALL_STATES)
        cfg = TARGET_CONFIG[target]
        df = loaded[target]
        ts = state_series(df, cfg, state)
        if ts.empty:
            st.info("No series available for this selection.")
        else:
            st.line_chart(ts.set_index("Year_num")[["state","national","traj_state","traj_nat"]])
            latest_row = ts.dropna(subset=["state"]).tail(1)
            if latest_row.empty:
                st.caption("No latest data point found for the state.")
            else:
                a = latest_row["state"].iloc[0]
                t_val = latest_row["traj_state"].iloc[0] if not pd.isna(latest_row["traj_state"].iloc[0]) else latest_row["traj_nat"].iloc[0]
                status = status_from_actual_vs_traj(a, t_val, lower_is_better=cfg["lower_is_better"])
                st.markdown(f"**Latest state status:** {render_traffic_cell(status)}", unsafe_allow_html=True)

elif page == "Playbooks":
    st.markdown("## Playbooks (auto-generated)")
    state = st.selectbox("Select state/territory", ALL_STATES)
    rows = []
    for key, cfg in TARGET_CONFIG.items():
        if key not in loaded: 
            continue
        dfk = loaded[key]
        la = latest_actual(dfk, cfg["measure_hint"])
        lt = latest_trajectory(dfk, cfg["measure_hint"])
        if not la or not lt:
            rows.append({"Target": key, "Name": cfg["label"], "Status":"Unknown", "State value": None, "National": None})
            continue
        y_a, da = la
        y_t, dt = lt
        a_state = pd.to_numeric(da[state], errors="coerce").dropna()
        a_nat = pd.to_numeric(da["Aust"], errors="coerce").dropna()
        t_state = pd.to_numeric(dt[state], errors="coerce").dropna()
        t_nat = pd.to_numeric(dt["Aust"], errors="coerce").dropna()
        a = a_state.iloc[0] if not a_state.empty else np.nan
        n = a_nat.iloc[0] if not a_nat.empty else np.nan
        t = (t_state.iloc[0] if not t_state.empty else (t_nat.iloc[0] if not t_nat.empty else np.nan))
        status = status_from_actual_vs_traj(a, t, lower_is_better=cfg["lower_is_better"])
        rows.append({"Target": key, "Name": cfg["label"], "Status": status, "State value": a, "National": n, "Trajectory": t, "Year": y_a})
    snap = pd.DataFrame(rows)
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("### Traffic-light snapshot")
        if not snap.empty:
            snap_disp = snap.copy()
            snap_disp["Status"] = snap_disp["Status"].apply(render_traffic_cell)
            st.write(snap_disp.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No data to display.")
    with col2:
        if pr_proxy is not None and "proxy_index_0_100" in pr_proxy.columns:
            val = pr_proxy.loc[pr_proxy["state"]==state.upper(), "proxy_index_0_100"]
            score = int(val.iloc[0]) if not val.empty else 0
            st.metric("Priority Reforms proxy", f"{score}/100")
        else:
            st.caption("Add `priority_reforms_proxy_by_state.csv` or `priority_reforms_signals.csv` to show PR proxy score.")
    st.write("---")
    st.write("### Export playbook")
    if not snap.empty:
        # CSV export
        st.download_button("Download CSV", data=snap.to_csv(index=False).encode("utf-8"), file_name=f"playbook_{state}.csv", mime="text/csv")
        # PDF export
        if REPORTLAB_OK:
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4)
            styles = getSampleStyleSheet()
            flow = [Paragraph(f"Playbook - {state}", styles["Title"]), Spacer(1, 12)]
            # Table
            tbl = Table([list(snap.columns)] + snap.astype(str).values.tolist(), repeatRows=1)
            tbl.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("FONTSIZE", (0,0), (-1,-1), 9),
                ("ALIGN", (0,0), (-1,0), "CENTER"),
            ]))
            flow.append(tbl)
            doc.build(flow)
            pdf_bytes = buf.getvalue()
            st.download_button("Download PDF", data=pdf_bytes, file_name=f"playbook_{state}.pdf", mime="application/pdf")
        else:
            st.caption("Install reportlab to enable PDF export (add to requirements.txt).")

elif page == "Map view":
    st.markdown("## Map view (state bubbles)")
    if not loaded:
        st.info("Upload datasets to use the map.")
    else:
        target = st.selectbox("Target", [k for k in TARGET_CONFIG.keys() if k in loaded], format_func=lambda k: f"{k} - {TARGET_CONFIG[k]['label']}")
        cfg = TARGET_CONFIG[target]
        pd_df = positive_deviance_latest(loaded[target], cfg)
        if pd_df.empty:
            st.info("No latest Actual found for this target.")
        else:
            map_df = map_dataframe_from_pd(pd_df)
            st.caption("Bubble size approximates z-score magnitude; position is a fixed centroid per state (not a true choropleth).")
            st.map(map_df.rename(columns={"lat":"latitude","lon":"longitude"}))
            st.dataframe(pd_df)

else:
    st.markdown("## About")
    st.write("""
CTG Multiplier Map - a minimalist dashboard for GovHack 2025.

- Uses official CtG CSVs (Targets 1,4,10,12,14,17; optional T2).
- Status = latest Actual vs Trajectory (green/amber/red).
- Positive deviance = state z-scores in the latest year (per target).
- Priority Reforms: load either a ready-made proxy CSV, or supply granular priority_reforms_signals.csv with state,signal,weight.
- Playbooks: export CSV + (optional) PDF.
""")
    st.info(DATA_HINT)
