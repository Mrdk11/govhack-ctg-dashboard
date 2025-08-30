
# CTG Multiplier Map (GovHack 2025)

A lightweight dashboard to explore Closing the Gap outcomes, find positive-deviant regions, and view a proxy for Priority Reforms activity.

## Files to include
Place these files in the repo root (or a `data/` folder):
- `app.py`
- `requirements.txt`
- CSV datasets from the Productivity Commission dashboard:
  - `ctg-202507-ctg01-healthy-lives-dataset.csv`
  - `ctg-202507-ctg04-early-years-dataset.csv`
  - `ctg-202507-ctg10-justice-dataset.csv`
  - `ctg-202507-ctg12-child-protection-dataset.csv`
  - `ctg-202507-ctg14-wellbeing-dataset.csv`
  - `ctg-202507-ctg17-digital-inclusion-dataset.csv`
- Optional:
  - `priority_reforms_proxy_by_state.csv` (Step 5 output)
  - `step4_t4_top10.csv` (Step 4 output, top AEDC performers)

## Run locally
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → New app → select repo/branch.
3. Main file path: `app.py`
4. Deploy — you’ll get a public URL.

## Notes
- Status logic = compare **Actual vs Trajectory** for latest year (Indigenous, All people, national values).
- The Priority Reforms proxy is a transparent, simple index built from documented 2025 actions; adjust as needed.
- To add T2 (Healthy birthweight) positive deviance, upload `ctg-202507-ctg02-healthy-birthweight-dataset.csv` and extend analysis similarly to T4.
