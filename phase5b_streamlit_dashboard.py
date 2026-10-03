"""
Phase 5b - Streamlit Dashboard
Run:  streamlit run phase5b_streamlit_dashboard.py
Requires Phase 5a FastAPI server running on port 8000 for live forecasts.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np                        # FIX: was missing import
import plotly.express as px
import plotly.graph_objects as go
import requests

# ── Paths (absolute, relative to this file) ───────────────────────────
ROOT    = os.path.dirname(os.path.abspath(__file__))
# FIX: was './data/dataco/...' which doesn't exist
DATACO_CSV  = os.path.join(ROOT, 'DataCo Smart Supply Chain',
                            'DataCoSupplyChainDataset.csv')
INVENTORY_CSV = os.path.join(ROOT, 'reports', 'inventory_plan.csv')
ROUTES_CSV    = os.path.join(ROOT, 'reports', 'route_summary.csv')
ROUTE_IMG     = os.path.join(ROOT, 'reports', 'route_optimization.png')
API_BASE      = "http://localhost:8002"

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(page_title="Supply Chain AI", page_icon="chain", layout="wide")
st.title("AI-Driven Supply Chain Optimization")

# ── Sidebar ───────────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Demand Forecast", "Route Optimizer", "Inventory Plan"]
)

# ── API status badge ──────────────────────────────────────────────────
try:
    health = requests.get(f"{API_BASE}/health", timeout=2).json()
    api_ok = health.get("status") == "ok"
except Exception:
    api_ok = False

st.sidebar.markdown("---")
if api_ok:
    st.sidebar.success("API server: Online")
else:
    st.sidebar.warning("API server: Offline  \n(run phase5a_fastapi_server.py)")

# ─────────────────────────────────────────────────────────────────────
# PAGE 1 – OVERVIEW
# ─────────────────────────────────────────────────────────────────────
if page == "Overview":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders",      "180,519", "+12%")
    col2.metric("Avg Delivery Days", "3.9",     "-0.5")
    col3.metric("On-Time Rate",      "67.3%",   "+4%")
    col4.metric("Forecast MAPE",     "9.2%",    "-3%")

    st.subheader("Monthly Revenue Trend")
    # FIX: use correct path
    if os.path.exists(DATACO_CSV):
        with st.spinner("Loading DataCo dataset ..."):
            df_plot = pd.read_csv(DATACO_CSV, encoding='latin-1',
                                  parse_dates=['order date (DateOrders)'])
        monthly = (df_plot.set_index('order date (DateOrders)')
                   .resample('ME')['Sales'].sum().reset_index())
        st.plotly_chart(
            px.line(monthly, x='order date (DateOrders)', y='Sales',
                    title='Monthly Revenue Trend',
                    labels={'order date (DateOrders)': 'Month', 'Sales': 'Revenue ($)'}),
            use_container_width=True
        )

        # Shipping mode breakdown
        st.subheader("Shipping Mode Distribution")
        ship = df_plot['Shipping Mode'].value_counts().reset_index()
        ship.columns = ['Shipping Mode', 'Count']
        st.plotly_chart(
            px.pie(ship, names='Shipping Mode', values='Count',
                   title='Orders by Shipping Mode', hole=0.4),
            use_container_width=True
        )
    else:
        st.error(f"DataCo CSV not found:\n{DATACO_CSV}")

# ─────────────────────────────────────────────────────────────────────
# PAGE 2 – DEMAND FORECAST
# ─────────────────────────────────────────────────────────────────────
elif page == "Demand Forecast":
    st.subheader("7-Day Demand Forecast (via LSTM API)")

    if not api_ok:
        st.error("FastAPI server is not running. Start it with:  \n"
                 "`python phase5a_fastapi_server.py`")
    else:
        col1, col2 = st.columns(2)
        product   = col1.text_input("Product Code", "Product_1359")
        warehouse = col2.selectbox("Warehouse",
                                   ["Whse_A", "Whse_B", "Whse_C", "Whse_J"])

        if st.button("Run Forecast"):
            # FIX: numpy was not imported before
            recent = list(np.random.randint(100, 500, 30).astype(float))
            try:
                resp = requests.post(
                    f"{API_BASE}/predict/demand",
                    json={"product_code": product, "warehouse": warehouse,
                          "horizon_days": 7, "recent_demand": recent},
                    timeout=10
                )
                if resp.ok:
                    fc     = resp.json()["forecast_7days"]
                    fc_df  = pd.DataFrame({"Day": range(1, 8), "Forecast": fc})
                    st.plotly_chart(
                        px.bar(fc_df, x="Day", y="Forecast",
                               title=f"7-Day Forecast for {product}",
                               color="Forecast",
                               color_continuous_scale="teal"),
                        use_container_width=True
                    )
                    st.success(f"Total 7-day forecast: {sum(fc):.0f} units")
                else:
                    # FIX: show API error detail instead of silently failing
                    st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is phase5a_fastapi_server.py running?")

# ─────────────────────────────────────────────────────────────────────
# PAGE 3 – ROUTE OPTIMIZER (was missing implementation entirely)
# ─────────────────────────────────────────────────────────────────────
elif page == "Route Optimizer":
    st.subheader("Vehicle Route Optimization (Phase 3 Results)")

    if os.path.exists(ROUTES_CSV):
        route_df = pd.read_csv(ROUTES_CSV)
        st.dataframe(route_df, use_container_width=True)

        fig = px.bar(route_df, x='vehicle', y='distance_km',
                     title='Distance per Vehicle (km)',
                     labels={'vehicle': 'Vehicle', 'distance_km': 'Distance (km)'},
                     color='distance_km', color_continuous_scale='reds')
        st.plotly_chart(fig, use_container_width=True)

        total_km = route_df['distance_km'].sum()
        st.metric("Total Fleet Distance", f"{total_km:,.1f} km")
    else:
        st.warning("route_summary.csv not found. Run Phase 3 first.")

    if os.path.exists(ROUTE_IMG):
        st.subheader("Route Map")
        st.image(ROUTE_IMG, caption="Greedy Nearest-Neighbour VRP – Brazil",
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# PAGE 4 – INVENTORY PLAN
# ─────────────────────────────────────────────────────────────────────
elif page == "Inventory Plan":
    st.subheader("SKU Inventory Plan (Phase 4 Results)")

    # FIX: use absolute path instead of './reports/inventory_plan.csv'
    if not os.path.exists(INVENTORY_CSV):
        st.error(f"inventory_plan.csv not found. Run Phase 4 first.\n{INVENTORY_CSV}")
        st.stop()

    inv = pd.read_csv(INVENTORY_CSV)

    abc_filter = st.multiselect("Filter by ABC Class", ["A", "B", "C"],
                                default=["A", "B"])
    filtered = inv[inv["ABC"].isin(abc_filter)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered SKUs",  f"{len(filtered):,}")
    col2.metric("Avg EOQ",        f"{filtered['EOQ'].mean():.0f} units")
    col3.metric("Avg Safety Stock", f"{filtered['safety_stock'].mean():.0f} units")

    st.dataframe(
        filtered[["Product_Code", "avg_daily_demand", "EOQ",
                  "safety_stock", "reorder_point", "ABC"]].head(50),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(filtered, x="EOQ", y="safety_stock",
                   color="ABC",
                   hover_data=["Product_Code"],
                   title="EOQ vs Safety Stock by ABC Class",
                   color_discrete_map={"A": "#e74c3c", "B": "#f39c12", "C": "#2ecc71"}),
        use_container_width=True
    )

    # Reorder point distribution
    st.plotly_chart(
        px.histogram(filtered, x="reorder_point", color="ABC", nbins=40,
                     title="Reorder Point Distribution",
                     color_discrete_map={"A": "#e74c3c", "B": "#f39c12", "C": "#2ecc71"}),
        use_container_width=True
    )