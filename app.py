
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

def first_notna(*vals):
    import pandas as pd, numpy as np
    for v in vals:
        if v is None:
            continue
        try:
            # treat pandas NA and numpy nan as missing
            if pd.notna(v) and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
        except Exception:
            # if it's a Series/Index etc. try to get scalar
            try:
                vv = v.iloc[0]
                if pd.notna(vv) and not (isinstance(vv, float) and np.isnan(vv)):
                    return float(vv)
            except Exception:
                continue
    return float("nan")

st.set_page_config(page_title="Closing the Gap – Multiplier Map", layout="wide")

DATA_HINT = """
Place these files alongside app.py (same folder):

- ctg-202507-ctg01-healthy-lives-dataset.csv
- ctg-202507-ctg04-early-years-dataset.csv
- ctg-202507-ctg10-justice-dataset.csv
- ctg-202507-ctg12-child-protection-dataset.csv
- ctg-202507-ctg14-wellbeing-dataset.csv
- ctg-202507-ctg17-digital-inclusion-dataset.csv
(optional, improves Playbooks)
- priority_reforms_proxy_by_state.csv
- step4_t4_top10.csv
"""


st.sidebar.title("CtG – Dashboard")
page = st.sidebar.radio("Navigate", ["Overview", "Regional drilldown", "Playbooks"])

def find_file(name):
    p = Path(name)
    if p.exists():
        return p
    # also try ./data
    p2 = Path("data") / name
    return p2 if p2.exists() else None

# ---- Load core datasets (Targets 1,4,10,12,14,17) ----
RAW_FILES = {
    "T1": "ctg-202507-ctg01-healthy-lives-dataset.csv",
    "T4": "ctg-202507-ctg04-early-years-dataset.csv",
    "T10": "ctg-202507-ctg10-justice-dataset.csv",
    "T12": "ctg-202507-ctg12-child-protection-dataset.csv",
    "T14": "ctg-202507-ctg14-wellbeing-dataset.csv",
    "T17": "ctg-202507-ctg17-digital-inclusion-dataset.csv",
}

def load_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except:
        return pd.read_csv(path, encoding="latin1")

loaded = {}
missing = []
for k, fname in RAW_FILES.items():
    fp = find_file(fname)
    if fp is None:
        missing.append(fname)
    else:
        loaded[k] = load_csv(fp)

if missing:
    st.warning("Missing files:\n" + "\n".join(f"- {m}" for m in missing))
    st.info(DATA_HINT)

# Optional proxies
pr_proxy_path = find_file("priority_reforms_proxy_by_state.csv")
pr_proxy = None
if pr_proxy_path:
    pr_proxy = pd.read_csv(pr_proxy_path)

t4_top10_path = find_file("step4_t4_top10.csv")
t4_top10 = None
if t4_top10_path:
    t4_top10 = pd.read_csv(t4_top10_path)

# ---- Helper functions ----
STATE_COLS = ["NSW","Vic","Qld","WA","SA","Tas","ACT","NT","Aust"]

def latest_actual(df, filter_measure_substr=None, ind_only=True, sex="All people"):
    x = df.copy()
    if filter_measure_substr:
        x = x[x["Measure"].str.contains(filter_measure_substr, case=False, na=False)]
    if ind_only:
        x = x[x["Indigenous_Status"].str.contains("Aboriginal and Torres Strait Islander people", na=False)]
    if sex:
        x = x[x["Sex"] == sex]
    # "Actual" rows are in Description3 for most tables
    if "Description3" in x.columns:
        x = x[x["Description3"] == "Actual"]
    # latest year where we have numbers
    x["Year_num"] = pd.to_numeric(x["Year"], errors="coerce")
    y = x.dropna(subset=["Year_num"]).sort_values("Year_num", ascending=False)
    if y.empty:
        return None
    latest_year = int(y["Year_num"].iloc[0])
    y = y[y["Year_num"] == latest_year]
    return latest_year, y

def latest_trajectory(df, filter_measure_substr=None, ind_only=True, sex="All people"):
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

def melt_values(df):
    v = df.melt(id_vars=[c for c in df.columns if c not in STATE_COLS],
                value_vars=[c for c in STATE_COLS if c in df.columns],
                var_name="region_code", value_name="value")
    v["value"] = pd.to_numeric(v["value"], errors="coerce")
    v = v.dropna(subset=["value"])
    return v

def status_from_actual_vs_traj(actual_val, traj_val, lower_is_better=False):
    # Simple status rules: compare actual to trajectory in latest year
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

TARGET_CONFIG = {
    "T1": {"label": "Healthy lives (incl. life expectancy)", "measure_hint": "life expectancy", "lower_is_better": False},
    "T4": {"label": "Early years (AEDC – all 5 domains)", "measure_hint": "developmentally on track", "lower_is_better": False},
    "T10":{"label": "Justice – adult imprisonment", "measure_hint": "imprisonment rate", "lower_is_better": True},
    "T12":{"label": "Children in out‑of‑home care", "measure_hint": "out-of-home care|out of home care", "lower_is_better": True},
    "T14":{"label": "Social & emotional wellbeing (suicide proxy)", "measure_hint": "suicide|psychological", "lower_is_better": True},
    "T17":{"label": "Digital inclusion / access", "measure_hint": "internet|online|digital|broadband|device", "lower_is_better": False},
}

def national_status_table():
    rows = []
    for key, cfg in TARGET_CONFIG.items():
        if key not in loaded:
            continue
        df = loaded[key]
        la = latest_actual(df, filter_measure_substr=cfg["measure_hint"])
        lt = latest_trajectory(df, filter_measure_substr=cfg["measure_hint"])
        if not la or not lt:
            rows.append({"Target": key, "Name": cfg["label"], "Year": None, "National": None, "Status": "Unknown"})
            continue
        y_a, da = la
        y_t, dt = lt
        # take national value 'Aust' (aggregate)
        nat_a = pd.to_numeric(da["Aust"], errors="coerce").dropna()
        nat_t = pd.to_numeric(dt["Aust"], errors="coerce").dropna()
        a = nat_a.iloc[0] if not nat_a.empty else np.nan
        t = nat_t.iloc[0] if not nat_t.empty else np.nan
        status = status_from_actual_vs_traj(a, t, lower_is_better=cfg["lower_is_better"])
        rows.append({"Target": key, "Name": cfg["label"], "Year": y_a, "National": a, "Trajectory": t, "Status": status})
    return pd.DataFrame(rows)

def state_series(df, cfg, state_code):
    x = df.copy()
    x = x[x["Measure"].str.contains(cfg["measure_hint"], case=False, na=False)]
    x = x[x["Indigenous_Status"].str.contains("Aboriginal and Torres Strait Islander people", na=False)]
    x = x[x["Sex"] == "All people"]

    # bail out early if the state column doesn't exist in the file
    if state_code not in x.columns:
        return pd.DataFrame()

    x["Year_num"] = pd.to_numeric(x["Year"], errors="coerce")

    # actuals
    xa = x[x.get("Description3", pd.Series(index=x.index, dtype=str)) == "Actual"].copy()
    xa = xa.dropna(subset=["Year_num"]).sort_values("Year_num")
    # make sure numeric
    for col in (state_code, "Aust"):
        if col in xa.columns:
            xa[col] = pd.to_numeric(xa[col], errors="coerce")

    df_state = xa[["Year_num", state_code]].rename(columns={state_code:"state"}) if state_code in xa.columns else pd.DataFrame(columns=["Year_num","state"])
    df_nat   = xa[["Year_num", "Aust"]].rename(columns={"Aust":"national"}) if "Aust" in xa.columns else pd.DataFrame(columns=["Year_num","national"])
    ts = df_state.merge(df_nat, on="Year_num", how="outer")

    # trajectory (may be missing per-state)
    xt = x[x.get("Description3", pd.Series(index=x.index, dtype=str)) == "Trajectory"].copy()
    xt["Year_num"] = pd.to_numeric(xt["Year"], errors="coerce")
    for col in (state_code, "Aust"):
        if col in xt.columns:
            xt[col] = pd.to_numeric(xt[col], errors="coerce")
    xt = xt[["Year_num", state_code, "Aust"]].rename(columns={state_code:"traj_state", "Aust":"traj_nat"}) if not xt.empty else pd.DataFrame(columns=["Year_num","traj_state","traj_nat"])

    ts = ts.merge(xt, on="Year_num", how="outer")
    return ts.sort_values("Year_num")


def render_traffic_cell(status):
    color = {"On track":"#1f8a70", "Not on track":"#e3a008", "Worsening":"#d64545", "Unknown":"#6b7280"}.get(status, "#6b7280")
    return f"<div style='background:{color};color:white;padding:4px 8px;border-radius:6px;text-align:center'>{status}</div>"

# ---------------- PAGES -----------------
if page == "Overview":
    st.markdown("## National Overview")
    st.caption("Traffic-light summary uses latest Actual vs Trajectory for each target (Indigenous, All people, national).")
    if loaded:
        table = national_status_table()
        if not table.empty:
            # Pretty render Status
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
    if pr_proxy is not None:
        st.bar_chart(pr_proxy.set_index("state")["proxy_index_0_100"])
        st.dataframe(pr_proxy)
    else:
        st.caption("Optional: add `priority_reforms_proxy_by_state.csv` for this chart.")

elif page == "Regional drilldown":
    st.markdown("## Regional Drilldown")
    if not loaded:
        st.info("Upload the CtG CSVs to enable drilldown.")
    else:
        target = st.selectbox("Select target", list(TARGET_CONFIG.keys()), format_func=lambda k: f"{k} – {TARGET_CONFIG[k]['label']}")
        state = st.selectbox("Select state/territory", ["NSW","Vic","Qld","WA","SA","Tas","ACT","NT"])
        cfg = TARGET_CONFIG[target]
        df = loaded[target]
        ts = state_series(df, cfg, state)
        if ts.empty:
            st.info("No series available for this selection.")
        else:
            st.line_chart(ts.set_index("Year_num")[["state","national","traj_state","traj_nat"]])
                        # Latest status for the state
            latest_row = ts.dropna(subset=["state"]).tail(1)
            if latest_row.empty:
                st.caption("No latest data point found for the state.")
            else:
                a = latest_row["state"].iloc[0]
                # robust fallback: traj_state -> traj_nat -> national
                t = first_notna(
                    latest_row["traj_state"].iloc[0] if "traj_state" in latest_row.columns else None,
                    latest_row["traj_nat"].iloc[0]   if "traj_nat"   in latest_row.columns else None,
                    latest_row["national"].iloc[0]   if "national"   in latest_row.columns else None,
                )
                status = status_from_actual_vs_traj(a, t, lower_is_better=cfg["lower_is_better"])
                st.markdown(f"**Latest state status:** {render_traffic_cell(status)}", unsafe_allow_html=True)

elif page == "Playbooks":
    st.markdown("## Playbooks (auto-generated)")
    state = st.selectbox("Select state/territory", ["NSW","Vic","Qld","WA","SA","Tas","ACT","NT"])
    # Build a compact snapshot
    rows = []
    for key, cfg in TARGET_CONFIG.items():
        if key not in loaded: 
            continue
        la = latest_actual(loaded[key], cfg["measure_hint"])
        lt = latest_trajectory(loaded[key], cfg["measure_hint"])
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
        if pr_proxy is not None:
            val = pr_proxy.loc[pr_proxy["state"]==state.upper(), "proxy_index_0_100"]
            score = int(val.iloc[0]) if not val.empty else 0
            st.metric("Priority Reforms proxy", f"{score}/100")
        else:
            st.caption("Add `priority_reforms_proxy_by_state.csv` to show PR proxy score.")
    st.write("---")
    st.write("### Export playbook (CSV)")
    if not snap.empty:
        st.download_button("Download playbook CSV", data=snap.to_csv(index=False).encode("utf-8"), file_name=f"playbook_{state}.csv", mime="text/csv")
    else:
        st.caption("No snapshot to export.")
