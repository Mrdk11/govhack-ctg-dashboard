
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from pathlib import Path

st.set_page_config(page_title="Closing the Gap – Multiplier Map (PDF fixed)", layout="wide")

# ---------- utilities ----------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def render_traffic_cell(status):
    color = {"On track":"#1f8a70", "Not on track":"#e3a008", "Worsening":"#d64545", "Unknown":"#6b7280"}.get(status, "#6b7280")
    return f"<div style='background:{color};color:white;padding:4px 8px;border-radius:6px;text-align:center'>{status}</div>"

def status_from_actual_vs_traj(actual_val, traj_val, lower_is_better=False):
    a = safe_float(actual_val); t = safe_float(traj_val)
    if np.isnan(a) or np.isnan(t):
        return "Unknown"
    if lower_is_better:
        if a <= t: return "On track"
        if a <= t * 1.1: return "Not on track"
        return "Worsening"
    else:
        if a >= t: return "On track"
        if a >= t * 0.9: return "Not on track"
        return "Worsening"

def find_file(name):
    p = Path(name)
    if p.exists(): return p
    p2 = Path("data") / name
    return p2 if p2.exists() else None

def load_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except:
        return pd.read_csv(path, encoding="latin1")

# ---------- data ----------
RAW_FILES = {
    "T1": "ctg-202507-ctg01-healthy-lives-dataset.csv",
    "T4": "ctg-202507-ctg04-early-years-dataset.csv",
    "T10": "ctg-202507-ctg10-justice-dataset.csv",
    "T12": "ctg-202507-ctg12-child-protection-dataset.csv",
    "T14": "ctg-202507-ctg14-wellbeing-dataset.csv",
    "T17": "ctg-202507-ctg17-digital-inclusion-dataset.csv",
}

TARGET_CONFIG = {
    "T1": {"label": "Healthy lives (incl. life expectancy)", "measure_hint": "life expectancy", "lower_is_better": False},
    "T4": {"label": "Early years (AEDC – all 5 domains)", "measure_hint": "developmentally on track", "lower_is_better": False},
    "T10":{"label": "Justice – adult imprisonment", "measure_hint": "imprisonment rate", "lower_is_better": True},
    "T12":{"label": "Children in out‑of‑home care", "measure_hint": "out-of-home care|out of home care", "lower_is_better": True},
    "T14":{"label": "Social & emotional wellbeing (suicide proxy)", "measure_hint": "suicide|psychological", "lower_is_better": True},
    "T17":{"label": "Digital inclusion / access", "measure_hint": "internet|online|digital|broadband|device", "lower_is_better": False},
}

loaded, missing = {}, []
for k,f in RAW_FILES.items():
    fp = find_file(f)
    if fp is None: missing.append(f)
    else: loaded[k] = load_csv(fp)

pr_proxy_path = find_file("priority_reforms_proxy_by_state.csv")
pr_proxy = pd.read_csv(pr_proxy_path) if pr_proxy_path else None

# ---------- helpers for latest ----------
def latest_actual(df, filter_measure_substr=None, ind_only=True, sex="All people"):
    x = df.copy()
    if filter_measure_substr:
        x = x[x["Measure"].str.contains(filter_measure_substr, case=False, na=False)]
    if ind_only:
        x = x[x["Indigenous_Status"].str.contains("Aboriginal and Torres Strait Islander people", na=False)]
    if sex: x = x[x["Sex"] == sex]
    if "Description3" in x.columns: x = x[x["Description3"] == "Actual"]
    x["Year_num"] = pd.to_numeric(x["Year"], errors="coerce")
    y = x.dropna(subset=["Year_num"]).sort_values("Year_num", ascending=False)
    if y.empty: return None
    latest = int(y["Year_num"].iloc[0])
    y = y[y["Year_num"] == latest]
    return latest, y

def latest_trajectory(df, filter_measure_substr=None, ind_only=True, sex="All people"):
    x = df.copy()
    if filter_measure_substr:
        x = x[x["Measure"].str.contains(filter_measure_substr, case=False, na=False)]
    if ind_only:
        x = x[x["Indigenous_Status"].str.contains("Aboriginal and Torres Strait Islander people", na=False)]
    if sex: x = x[x["Sex"] == sex]
    if "Description3" in x.columns: x = x[x["Description3"] == "Trajectory"]
    x["Year_num"] = pd.to_numeric(x["Year"], errors="coerce")
    y = x.dropna(subset=["Year_num"]).sort_values("Year_num", ascending=False)
    if y.empty: return None
    latest = int(y["Year_num"].iloc[0])
    y = y[y["Year_num"] == latest]
    return latest, y

# ---------- UI ----------
st.title("Closing the Gap – Playbook PDF (fixed)")
if missing:
    st.warning("Missing files:\n" + "\n".join(f"- {m}" for m in missing))

state = st.selectbox("Select state/territory", ["NSW","Vic","Qld","WA","SA","Tas","ACT","NT"])

rows = []
for key, cfg in TARGET_CONFIG.items():
    if key not in loaded: 
        continue
    la = latest_actual(loaded[key], cfg["measure_hint"])
    lt = latest_trajectory(loaded[key], cfg["measure_hint"])
    if not la or not lt:
        rows.append({"Target": key, "Name": cfg["label"], "Status":"Unknown", "State value": None, "National": None, "Trajectory": None, "Year": None})
        continue
    y_a, da = la
    y_t, dt = lt
    a_state = pd.to_numeric(da.get(state, pd.Series(dtype=float)), errors="coerce").dropna()
    a_nat = pd.to_numeric(da.get("Aust", pd.Series(dtype=float)), errors="coerce").dropna()
    t_state = pd.to_numeric(dt.get(state, pd.Series(dtype=float)), errors="coerce").dropna()
    t_nat = pd.to_numeric(dt.get("Aust", pd.Series(dtype=float)), errors="coerce").dropna()
    a = a_state.iloc[0] if not a_state.empty else np.nan
    n = a_nat.iloc[0] if not a_nat.empty else np.nan
    t = (t_state.iloc[0] if not t_state.empty else (t_nat.iloc[0] if not t_nat.empty else np.nan))
    status = status_from_actual_vs_traj(a, t, lower_is_better=cfg["lower_is_better"])
    rows.append({"Target": key, "Name": cfg["label"], "Status": status, "State value": a, "National": n, "Trajectory": t, "Year": y_a})
snap = pd.DataFrame(rows)

if not snap.empty:
    snap_disp = snap.copy()
    snap_disp["Status"] = snap_disp["Status"].apply(render_traffic_cell)
    st.write(snap_disp.to_html(escape=False, index=False), unsafe_allow_html=True)

# PDF generation (robust)
def build_playbook_pdf(state_code, snapshot_df, proxy_score=None):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Closing the Gap – Playbook: {state_code}", styles["Title"]))
    story.append(Spacer(1, 12))
    if proxy_score is not None:
        story.append(Paragraph(f"Priority Reforms proxy score: <b>{proxy_score}/100</b>", styles["Heading3"]))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Traffic-light snapshot", styles["Heading3"]))

    headers = ["Target", "Name", "Status", "State value", "National", "Trajectory", "Year"]
    data = [headers]
    for _, r in snapshot_df.iterrows():
        data.append([str(r.get("Target","")), str(r.get("Name","")), str(r.get("Status","")),
                     "" if pd.isna(r.get("State value")) else f"{r.get('State value'):.2f}",
                     "" if pd.isna(r.get("National")) else f"{r.get('National'):.2f}",
                     "" if pd.isna(r.get("Trajectory")) else f"{r.get('Trajectory'):.2f}",
                     str(r.get("Year",""))])

    tbl = Table(data, colWidths=[50, 210, 80, 80, 80, 80, 40])
    style = TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f3f4f6")),
                        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
                        ("ALIGN", (0,0), (-1,-1), "CENTER"),
                        ("ALIGN", (1,1), (1,-1), "LEFT"),
                        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                        ("FONTSIZE", (0,0), (-1,0), 10),
                        ("FONTSIZE", (0,1), (-1,-1), 9),
                        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb"))])
    # color status column
    for i in range(1, len(data)):
        s = data[i][2]
        c = {"On track": colors.green, "Not on track": colors.orange, "Worsening": colors.red}.get(s, colors.grey)
        style.add("TEXTCOLOR", (2,i), (2,i), c)
        style.add("FONTNAME", (2,i), (2,i), "Helvetica-Bold")
    tbl.setStyle(style)
    story.append(tbl)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Notes", styles["Heading3"]))
    story.append(Paragraph("• Status is computed from latest Actual vs Trajectory (Indigenous, All people).", styles["BodyText"]))
    story.append(Paragraph("• Proxy score is a simple index derived from the Commonwealth 2025 Implementation Actions table.", styles["BodyText"]))
    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf

st.write("---")
st.write("### Export playbook PDF")
pr_val = None
if pr_proxy is not None:
    v = pr_proxy.loc[pr_proxy["state"]==state.upper(), "proxy_index_0_100"]
    pr_val = int(v.iloc[0]) if not v.empty else None

if not snap.empty:
    try:
        pdf_bytes = build_playbook_pdf(state, snap, proxy_score=pr_val)
        if not pdf_bytes or len(pdf_bytes) == 0:
            st.error("PDF is empty. Please try again or check data.")
        else:
            st.download_button("Download Playbook PDF",
                               data=pdf_bytes,
                               file_name=f"playbook_{state}.pdf",
                               mime="application/pdf",
                               key=f"dl_{state}")
    except Exception as e:
        st.exception(e)
else:
    st.caption("No snapshot to export.")
