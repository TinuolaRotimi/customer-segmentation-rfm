from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

print("Loading and processing data...")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
df = pd.read_excel(DATA_URL, engine="openpyxl")
df = df.drop_duplicates()
df = df.dropna(subset=["CustomerID"])
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df = df[~df["InvoiceNo"].str.startswith("C")]
df = df[df["Quantity"] > 0]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate"])
df["Total_Transaction_Value"] = df["Quantity"] * df["UnitPrice"]
df = df[df["Total_Transaction_Value"] > 0]

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Total_Transaction_Value", "sum")
).reset_index()

X = rfm[["Recency", "Frequency", "Monetary"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
rfm["Segment_Number"] = kmeans.fit_predict(X_scaled)

segment_labels = {0: "Recent Customers", 1: "Lost", 2: "Loyal Customers", 3: "Loyal Customers", 4: "Champions"}
rfm["Segment_Label"] = rfm["Segment_Number"].map(segment_labels)

recommendations = {
    "Recent Customers": "Deploy a welcome email series introducing the product catalog and brand story. Build early engagement and establish a second-purchase habit.",
    "Loyal Customers": "Launch a referral program and personalized cross-sell bundles based on purchase history. Increase average order value while leveraging their existing trust.",
    "Champions": "Enroll in a VIP loyalty tier with early access to new product launches and exclusive perks. Reinforce brand advocacy and protect from competitor poaching.",
    "Lost": "Deprioritize active marketing spend and retain only in low-cost newsletter lists. Reallocate marketing budget toward higher-yield segments.",
}

overall = {
    "avg_recency": round(rfm["Recency"].mean(), 1),
    "avg_frequency": round(rfm["Frequency"].mean(), 1),
    "avg_monetary": round(rfm["Monetary"].mean(), 2),
    "total_customers": len(rfm),
    "total_revenue": round(rfm["Monetary"].sum(), 2),
}

print(f"Model ready. {len(rfm)} customers segmented into 5 clusters.")

PAGE_CSS = """
:root{--bg:#0a0a0f;--card:#12121a;--border:#252540;--text:#e8e8f0;--muted:#9898b0;--accent:#6366f1;--green:#22c55e;--yellow:#eab308;--red:#ef4444;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;line-height:1.6;}
.wrap{max-width:900px;margin:0 auto;padding:40px 24px;}
header{border-bottom:1px solid var(--border);padding-bottom:24px;margin-bottom:32px;}
h1{font-size:2rem;font-weight:700;margin-bottom:6px;}
.sub{color:var(--muted);font-size:14px;}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:28px 0;}
label{font-family:monospace;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);}
select{background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:10px 14px;font-size:15px;min-width:240px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:28px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;}
.card .k{font-family:monospace;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:8px;}
.card .v{font-family:monospace;font-size:28px;font-weight:700;}
.badge{display:inline-block;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;}
.badge.champions{background:rgba(34,197,94,0.15);color:var(--green);border:1px solid rgba(34,197,94,0.3);}
.badge.loyal{background:rgba(99,102,241,0.15);color:var(--accent);border:1px solid rgba(99,102,241,0.3);}
.badge.recent{background:rgba(234,179,8,0.15);color:var(--yellow);border:1px solid rgba(234,179,8,0.3);}
.badge.lost{background:rgba(239,68,68,0.15);color:var(--red);border:1px solid rgba(239,68,68,0.3);}
.rec{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;margin-top:20px;}
.rec h3{font-size:14px;margin-bottom:10px;}
.rec p{color:var(--muted);font-size:14px;}
table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px;}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--border);}
th{font-family:monospace;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);}
footer{margin-top:48px;padding-top:20px;border-top:1px solid var(--border);color:var(--muted);font-size:12px;text-align:center;}
"""

@app.route("/")
def index():
    customers = sorted(rfm["CustomerID"].astype(int).tolist())
    selected = request.args.get("customer", str(customers[0]))
    selected_int = int(float(selected))
    
    row = rfm[rfm["CustomerID"] == selected_int]
    if row.empty:
        selected_int = customers[0]
        row = rfm[rfm["CustomerID"] == selected_int]
    
    row = row.iloc[0]
    label = row["Segment_Label"]
    badge_class = {"Champions": "champions", "Loyal Customers": "loyal", "Recent Customers": "recent", "Lost": "lost"}.get(label, "")
    rec = recommendations.get(label, "")
    
    options = "".join(f'<option value="{c}"{" selected" if c == selected_int else ""}>{c}</option>' for c in customers[:500])
    
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Customer Segmentation - RFM Analysis</title>
<style>{PAGE_CSS}</style>
</head><body><div class="wrap">
<header>
  <h1>Customer Segmentation Analysis</h1>
  <p class="sub">RFM (Recency, Frequency, Monetary) model with KMeans clustering on real e-commerce transaction data. 4,338 customers across 5 behavioral segments.</p>
</header>
<form method="get" class="controls">
  <label for="customer">Customer ID</label>
  <select id="customer" name="customer" onchange="this.form.submit()">{options}</select>
</form>
<div class="grid">
  <div class="card"><div class="k">Segment</div><div class="v" style="font-size:18px;"><span class="badge {badge_class}">{label}</span></div></div>
  <div class="card"><div class="k">Recency</div><div class="v">{int(row['Recency'])} <span style="font-size:14px;color:var(--muted);">days</span></div></div>
  <div class="card"><div class="k">Frequency</div><div class="v">{int(row['Frequency'])} <span style="font-size:14px;color:var(--muted);">orders</span></div></div>
  <div class="card"><div class="k">Monetary</div><div class="v" style="font-size:22px;">${row['Monetary']:,.2f}</div></div>
</div>
<div class="rec"><h3>Strategic Recommendation</h3><p>{rec}</p></div>
<table>
  <tr><th>Metric</th><th>This Customer</th><th>Average</th></tr>
  <tr><td>Recency (days)</td><td>{int(row['Recency'])}</td><td>{overall['avg_recency']}</td></tr>
  <tr><td>Frequency (orders)</td><td>{int(row['Frequency'])}</td><td>{overall['avg_frequency']}</td></tr>
  <tr><td>Monetary (revenue)</td><td>${row['Monetary']:,.2f}</td><td>${overall['avg_monetary']:,.2f}</td></tr>
</table>
<footer>Data: UCI Online Retail Dataset | Model: KMeans Clustering (k=5) | {overall['total_customers']:,} customers analyzed</footer>
</div></body></html>"""

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
