"""
Phase 3 – Route Optimization (Nearest-Neighbour VRP)
Reads Olist geo + order data; simulates a Vehicle Routing Problem
using a greedy nearest-neighbour heuristic and saves route metrics.
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving plots
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT    = os.path.dirname(os.path.abspath(__file__))
DATA_OR = os.path.join(ROOT, 'Olist Brazilian E-Commerce')
REPORTS = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS, exist_ok=True)

print("=" * 60)
print("PHASE 3 - ROUTE OPTIMISATION")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────
orders    = pd.read_csv(os.path.join(DATA_OR, 'olist_orders_dataset.csv'))
items     = pd.read_csv(os.path.join(DATA_OR, 'olist_order_items_dataset.csv'))
customers = pd.read_csv(os.path.join(DATA_OR, 'olist_customers_dataset.csv'))
geo       = pd.read_csv(os.path.join(DATA_OR, 'olist_geolocation_dataset.csv'))
sellers   = pd.read_csv(os.path.join(DATA_OR, 'olist_sellers_dataset.csv'))

print(f"  Orders: {orders.shape}  Items: {items.shape}  Geo: {geo.shape}")

# ── Aggregate demand per customer zip ─────────────────────────────────
merged = (orders
    .merge(customers[['customer_id', 'customer_zip_code_prefix']], on='customer_id')
    .merge(items[['order_id', 'price']].groupby('order_id')['price'].sum().reset_index(),
           on='order_id')
)

zip_demand = (merged
    .groupby('customer_zip_code_prefix')
    .agg(total_demand=('price', 'sum'), n_orders=('order_id', 'count'))
    .reset_index()
)

# Average geo coords per zip
geo_avg = (geo
    .groupby('geolocation_zip_code_prefix')
    .agg(lat=('geolocation_lat', 'mean'), lng=('geolocation_lng', 'mean'))
    .reset_index()
    .rename(columns={'geolocation_zip_code_prefix': 'customer_zip_code_prefix'})
)

stops = zip_demand.merge(geo_avg, on='customer_zip_code_prefix').dropna()
stops = stops[stops['total_demand'] > 0].copy()
print(f"  Stop locations with geo: {len(stops):,}")

# ── Vectorized Haversine distance matrix (km) ────────────────────────
# Pre-compute once → O(1) lookup during NN search instead of O(n) Python calls
def haversine_matrix(lats, lons):
    """Returns (n x n) distance matrix in km using NumPy broadcasting."""
    R      = 6371.0
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    dlat   = lats_r[:, None] - lats_r[None, :]
    dlon   = lons_r[:, None] - lons_r[None, :]
    a      = np.sin(dlat / 2)**2 + (
             np.cos(lats_r[:, None]) * np.cos(lats_r[None, :]) * np.sin(dlon / 2)**2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ── Greedy Nearest-Neighbour heuristic ───────────────────────────────
TOP_N      = 200
N_VEHICLES = 5
sample     = stops.nlargest(TOP_N, 'total_demand').reset_index(drop=True)

# Index 0 = depot (centroid), indices 1..TOP_N = delivery stops
depot_lat = sample['lat'].mean()
depot_lng = sample['lng'].mean()
print(f"  Depot (centroid): ({depot_lat:.3f}, {depot_lng:.3f})")
print(f"  Vehicles: {N_VEHICLES}  |  Stops: {TOP_N}")

all_lats = np.concatenate([[depot_lat], sample['lat'].values])
all_lons = np.concatenate([[depot_lng], sample['lng'].values])

print("  Building distance matrix … ", end='', flush=True)
dist_matrix = haversine_matrix(all_lats, all_lons)  # shape (TOP_N+1, TOP_N+1)
print("done.")

# Node 0 = depot; nodes 1..TOP_N = stops
unvisited  = set(range(1, TOP_N + 1))
routes     = [[] for _ in range(N_VEHICLES)]
total_dist = [0.0] * N_VEHICLES
cur_node   = [0] * N_VEHICLES          # all vehicles start at depot

while unvisited:
    for v in range(N_VEHICLES):
        if not unvisited:
            break
        uv_list = list(unvisited)
        # Nearest unvisited node from current position (single array lookup)
        dists    = dist_matrix[cur_node[v], uv_list]
        best_pos = int(np.argmin(dists))
        best_node = uv_list[best_pos]

        total_dist[v] += dists[best_pos]
        routes[v].append(best_node)
        cur_node[v] = best_node
        unvisited.discard(best_node)

# Return-to-depot leg
for v in range(N_VEHICLES):
    if routes[v]:
        total_dist[v] += dist_matrix[cur_node[v], 0]

route_summary = pd.DataFrame({
    'vehicle':     range(1, N_VEHICLES + 1),
    'stops':       [len(r) for r in routes],
    'distance_km': [round(d, 1) for d in total_dist]
})
print(f"\n  Route Summary:\n{route_summary.to_string(index=False)}")
print(f"  Total fleet distance: {sum(total_dist):.1f} km")

# ── Plot routes ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab10.colors
for v in range(N_VEHICLES):
    lats = [depot_lat] + [sample.loc[n - 1, 'lat'] for n in routes[v]] + [depot_lat]
    lngs = [depot_lng] + [sample.loc[n - 1, 'lng'] for n in routes[v]] + [depot_lng]
    ax.plot(lngs, lats, '-o', color=colors[v], markersize=3,
            label=f'Vehicle {v+1} ({total_dist[v]:.0f} km)', alpha=0.7)
ax.plot(depot_lng, depot_lat, 'k*', markersize=15, label='Depot')
ax.set_title('Greedy Nearest-Neighbour VRP – Brazil Deliveries')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.legend(fontsize=8)
plt.tight_layout()
plot_path = os.path.join(REPORTS, 'route_optimization.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"  Route plot saved -> {plot_path}")

route_summary.to_csv(os.path.join(REPORTS, 'route_summary.csv'), index=False)
print("\nPhase 3 complete.")