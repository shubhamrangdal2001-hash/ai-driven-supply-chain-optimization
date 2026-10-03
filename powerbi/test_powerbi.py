"""
Validate Power BI project files before opening in Desktop.
Run: python powerbi/test_powerbi.py
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBI_DATA = os.path.join(ROOT, "reports", "powerbi")
POWERBI = os.path.join(ROOT, "powerbi")
PBIP = os.path.join(POWERBI, "SupplyChainAnalytics.pbip")
MODEL = os.path.join(POWERBI, "SupplyChainAnalytics.SemanticModel", "model.bim")
PAGES = os.path.join(POWERBI, "SupplyChainAnalytics.Report", "definition", "pages")

REQUIRED_CSV = [
    "fact_orders.csv",
    "dashboard_kpis.csv",
    "monthly_trend.csv",
    "category_kpis.csv",
    "inventory_plan.csv",
]

MIN_ROWS = {
    "fact_orders.csv": 180_000,
    "monthly_trend.csv": 30,
    "category_kpis.csv": 40,
}


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    print("=" * 60)
    print("POWER BI PROJECT TEST")
    print("=" * 60)

    if not os.path.exists(PBIP):
        fail(f"Missing PBIP: {PBIP}")
    ok("SupplyChainAnalytics.pbip exists")
    with open(PBIP, encoding="utf-8") as f:
        pbip = json.load(f)
    artifacts = pbip.get("artifacts", [])
    if not any("report" in a for a in artifacts):
        fail("PBIP must reference the Report artifact")
    ok("PBIP references report (model via definition.pbir)")
    pbir = os.path.join(POWERBI, "SupplyChainAnalytics.Report", "definition.pbir")
    with open(pbir, encoding="utf-8") as f:
        ref = json.load(f)
    model_path = ref.get("datasetReference", {}).get("byPath", {}).get("path", "")
    if "SemanticModel" not in model_path:
        fail(f"definition.pbir missing semantic model path: {model_path}")
    ok("Report links semantic model by path")

    if not os.path.exists(MODEL):
        fail(f"Missing model.bim: {MODEL}")
    with open(MODEL, encoding="utf-8") as f:
        bim = json.load(f)
    tables = [t["name"] for t in bim["model"]["tables"]]
    ok(f"model.bim has {len(tables)} tables")

    for csv_name in REQUIRED_CSV:
        path = os.path.join(PBI_DATA, csv_name)
        if not os.path.exists(path):
            fail(f"Missing data: {path}")
        rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        min_r = MIN_ROWS.get(csv_name, 1)
        if rows < min_r:
            fail(f"{csv_name} has {rows} rows (need >={min_r})")
        ok(f"{csv_name}: {rows:,} rows")

    cert = os.path.join(PBI_DATA, "REAL_DATA_CERTIFICATE.json")
    if not os.path.exists(cert):
        fail("Missing REAL_DATA_CERTIFICATE.json — run powerbi_data_prep.py")
    ok("Real data certificate present")

    visuals = list(os.path.join(r, f) for r, _, fs in os.walk(PAGES) for f in fs if f == "visual.json")
    if len(visuals) < 10:
        fail(f"Only {len(visuals)} visuals found — run build_powerbi_desktop.py")
    ok(f"{len(visuals)} dashboard visuals defined")

    pages_json = os.path.join(PAGES, "pages.json")
    with open(pages_json, encoding="utf-8") as f:
        pages = json.load(f)
    ok(f"{len(pages['pageOrder'])} report pages: {', '.join(pages['pageOrder'])}")
    slug_re = re.compile(r"^[\w-]+$")
    for page_id in pages["pageOrder"]:
        page_json = os.path.join(PAGES, page_id, "page.json")
        if not os.path.exists(page_json):
            fail(f"Missing page.json for {page_id}")
        with open(page_json, encoding="utf-8") as f:
            page = json.load(f)
        if page.get("name") != page_id:
            fail(f"Page folder/name mismatch: {page_id}")
        if not slug_re.match(page_id):
            fail(f"Invalid page id (Desktop ignores): {page_id}")
        if "visualContainers" in page:
            fail(f"page.json must not use visualContainers: {page_id}")
    ok("Page folders match PBIR naming rules")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nOpen in Power BI DESKTOP (not the website):")
    print(f"  {os.path.join(ROOT, 'OPEN_POWERBI.bat')}")
    print("\nAfter Desktop opens:")
    print("  1. Home -> Refresh")
    print("  2. Wait for refresh to finish (180k+ order rows)")
    print("  3. Click page tabs: Executive | Logistics | Inventory | Demand")
    print("\nDo NOT use app.powerbi.com or Publish — local CSV data only.")


if __name__ == "__main__":
    main()
