"""
Build a Power BI Desktop-compatible project (PBIP + model.bim).
Uses model.bim (TMSL) instead of TMDL — opens without preview features.
Run: python build_powerbi_desktop.py
"""

import json
import os
import shutil
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POWERBI = os.path.dirname(os.path.abspath(__file__))
PBI_DATA_DIR = os.path.join(ROOT, "reports", "powerbi")
PBI_DATA = PBI_DATA_DIR.replace("\\", "\\\\")

SM = os.path.join(POWERBI, "SupplyChainAnalytics.SemanticModel")
REPORT = os.path.join(POWERBI, "SupplyChainAnalytics.Report")
TMDL_BACKUP = os.path.join(SM, "definition_tmdl_backup")

INT_COLS = {
    "Order_Year", "Order_Month", "Order_Quarter", "Day_of_Week", "Scheduled_Days",
    "Actual_Days", "Is_Late", "Late_Risk", "Quantity", "On_Time", "DateKey",
    "Year", "Month", "Quarter", "Day_of_Week", "Is_Weekend", "Orders", "Units_Sold",
    "vehicle", "stops", "Days_Active", "Total_Orders",
}
TEXT_COLS = {
    "Category", "Department", "Product", "Customer_Segment", "Market", "Region",
    "Country", "Shipping_Mode", "Delivery_Status", "Order_Status", "KPI", "Metric",
    "Warehouse", "SKU", "ABC_Class", "State", "Source", "Unit", "Format", "Month_Year",
}
DOUBLE_COLS = {
    "Delay_Days", "Revenue", "Profit", "Discount_Rate", "Profit_Ratio", "Profit_Margin",
    "On_Time_Pct", "Avg_Delay_Days", "Avg_Order_Value", "Avg_Discount", "Total_Demand",
    "Avg_Daily_Demand", "distance_km", "Avg_Delivery_Days", "Median_Delivery_Days",
    "P90_Delivery_Days", "EOQ", "safety_stock", "reorder_point", "annual_demand",
    "Total_Revenue", "Total_Profit", "On_Time_Rate_Pct",
}
DATE_COLS = {"Order_Date", "Date"}


def infer_dtype(col_name: str) -> str:
    if col_name in TEXT_COLS:
        return "string"
    if col_name in DATE_COLS or col_name.endswith("_Date"):
        return "dateTime"
    if col_name in INT_COLS:
        return "int64"
    if col_name in DOUBLE_COLS or col_name == "Value":
        return "double"
    lower = col_name.lower()
    if any(k in lower for k in ("revenue", "profit", "demand", "delay", "margin", "cost", "pct", "prob", "reduction", "overstock", "stockout", "avg_", "total_", "distance", "eoq", "std_", "cumulative")):
        return "double"
    # Match whole name segments only — avoid "count" matching inside "country".
    int_tokens = ("orders", "units", "quantity", "count", "stops", "vehicle", "iterations", "days_active", "skus")
    if lower in int_tokens or any(lower.endswith(f"_{t}") for t in int_tokens):
        return "int64"
    if col_name in ("Year", "Month", "Quarter", "Day_of_Week", "Is_Weekend", "Rows"):
        return "int64"
    return "string"


def columns_from_csv(filename: str) -> list[dict]:
    path = os.path.join(PBI_DATA_DIR, filename)
    with open(path, encoding="utf-8") as f:
        headers = f.readline().strip().split(",")
    cols = []
    for h in headers:
        dtype = infer_dtype(h)
        col_def: dict = {"name": h, "dataType": dtype, "sourceColumn": h}
        if dtype == "dateTime":
            col_def["formatString"] = "Short Date"
        elif dtype == "double":
            col_def["formatString"] = "#,0.00"
        elif dtype == "int64":
            col_def["formatString"] = "0"
        cols.append(col_def)
    return cols


def _m_type(dtype: str) -> str:
    if dtype == "dateTime":
        return "type date"
    if dtype == "int64":
        return "Int64.Type"
    if dtype == "double":
        return "type number"
    return "type text"


def m_csv(filename: str) -> list[str]:
    cols = columns_from_csv(filename)
    typed = [c for c in cols if _m_type(c["dataType"]) != "type text"]
    # Hardcode path — PBI_DataPath parameter causes cyclic-reference errors on Refresh in PBIP.
    csv_path = f"{PBI_DATA}\\\\{filename}"
    lines = [
        "let",
        f'    Source = Csv.Document(File.Contents("{csv_path}"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),',
        '    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true])',
    ]
    if typed:
        pairs = ", ".join(
            f'{{"{c["name"]}", {_m_type(c["dataType"])}}}' for c in typed
        )
        lines[2] += ","
        lines.append(
            f'    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers", {{{pairs}}})'
        )
        lines.append("in")
        lines.append('    #"Changed Type"')
    else:
        lines.append("in")
        lines.append('    #"Promoted Headers"')
    return lines


def table(name: str, filename: str, measures: list[dict] | None = None) -> dict:
    t: dict = {
        "name": name,
        "columns": columns_from_csv(filename),
        "partitions": [
            {
                "name": name,
                "mode": "import",
                "source": {"type": "m", "expression": m_csv(filename)},
            }
        ],
    }
    if measures:
        t["measures"] = measures
    return t


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)


VISUAL_SCHEMA = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.7.0/schema.json"
)
PAGE_SCHEMA = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.1.0/schema.json"
)
FONT_STACK = "''Segoe UI', wf_segoe-ui_normal, helvetica, arial, sans-serif'"


def _col(entity: str, prop: str) -> dict:
    return {
        "field": {
            "Column": {
                "Expression": {"SourceRef": {"Entity": entity}},
                "Property": prop,
            }
        },
        "queryRef": f"{entity}.{prop}",
        "nativeQueryRef": prop,
    }


def _meas(entity: str, prop: str) -> dict:
    return {
        "field": {
            "Measure": {
                "Expression": {"SourceRef": {"Entity": entity}},
                "Property": prop,
            }
        },
        "queryRef": f"{entity}.{prop}",
        "nativeQueryRef": prop,
    }


def _filter_config(query_state: dict, prefix: str) -> dict | None:
    filters: list[dict] = []
    for idx, bucket in enumerate(query_state.values(), start=1):
        for proj in bucket.get("projections", []):
            field = proj["field"]
            filters.append(
                {
                    "name": f"{prefix}{idx:02d}",
                    "field": json.loads(json.dumps(field)),
                    "type": "Advanced" if "Aggregation" in field else "Categorical",
                }
            )
    return {"filters": filters} if filters else None


def _query_block(query_state: dict) -> dict:
    block: dict = {"queryState": query_state}
    cat = query_state.get("Category") or query_state.get("Values")
    if cat and cat.get("projections"):
        field = json.loads(json.dumps(cat["projections"][0]["field"]))
        block["sortDefinition"] = {
            "sort": [{"field": field, "direction": "Ascending"}],
            "isDefaultSort": True,
        }
    return block


def _chart_objects() -> dict:
    font = {
        "fontSize": {"expr": {"Literal": {"Value": "8D"}}},
        "fontFamily": {"expr": {"Literal": {"Value": FONT_STACK}}},
    }
    return {
        "categoryAxis": [{"properties": font}],
        "valueAxis": [{"properties": font}],
        "labels": [{"properties": font}],
        "legend": [{"properties": font}],
    }


def _visual(
    vid: str,
    x: int,
    y: int,
    w: int,
    h: int,
    vtype: str,
    query_state: dict,
    z: int = 1000,
    title: str | None = None,
) -> dict:
    z_val = z if z >= 1000 else z * 1000
    visual: dict = {
        "visualType": vtype,
        "query": _query_block(query_state),
        "drillFilterOtherVisuals": True,
        "objects": _chart_objects(),
    }
    if title:
        visual["visualContainerObjects"] = {
            "title": [
                {
                    "properties": {
                        "text": {"expr": {"Literal": {"Value": f"'{title}'"}}},
                        "fontSize": {"expr": {"Literal": {"Value": "8D"}}},
                        "fontFamily": {"expr": {"Literal": {"Value": FONT_STACK}}},
                    }
                }
            ]
        }
    spec: dict = {
        "$schema": VISUAL_SCHEMA,
        "name": vid,
        "position": {"x": x, "y": y, "z": z_val, "height": h, "width": w, "tabOrder": z_val},
        "visual": visual,
    }
    filters = _filter_config(query_state, vid[-6:])
    if filters:
        spec["filterConfig"] = filters
    return spec


def _col_active(entity: str, prop: str) -> dict:
    p = _col(entity, prop)
    p["active"] = True
    return p


def _card_col(vid: str, x: int, y: int, w: int, h: int, entity: str, column: str, z: int = 1000, title: str | None = None) -> dict:
    return _visual(
        vid, x, y, w, h, "card",
        {"Values": {"projections": [_col(entity, column)]}},
        z, title or column.replace("_", " "),
    )


def _card_meas(vid: str, x: int, y: int, w: int, h: int, entity: str, measure: str, z: int = 1000) -> dict:
    return _visual(
        vid, x, y, w, h, "card",
        {"Values": {"projections": [_meas(entity, measure)]}},
        z, measure,
    )


def _line(vid: str, x: int, y: int, w: int, h: int, cat_e: str, cat_c: str, val_e: str, val_c: str, z: int = 0) -> dict:
    return _visual(
        vid, x, y, w, h, "lineChart",
        {
            "Category": {"projections": [_col_active(cat_e, cat_c)]},
            "Y": {"projections": [_col(val_e, val_c)]},
        },
        z,
    )


def _bar(vid: str, x: int, y: int, w: int, h: int, cat_e: str, cat_c: str, val_e: str, val_c: str, z: int = 0) -> dict:
    return _visual(
        vid, x, y, w, h, "barChart",
        {
            "Category": {"projections": [_col_active(cat_e, cat_c)]},
            "Y": {"projections": [_col(val_e, val_c)]},
        },
        z,
    )


def _donut(vid: str, x: int, y: int, w: int, h: int, cat_e: str, cat_c: str, val_e: str, val_c: str, z: int = 0) -> dict:
    return _visual(
        vid, x, y, w, h, "donutChart",
        {"Category": {"projections": [_col(cat_e, cat_c)]}, "Y": {"projections": [_col(val_e, val_c)]}},
        z,
    )


def _table(vid: str, x: int, y: int, w: int, h: int, fields: list[tuple[str, str]], z: int = 0) -> dict:
    return _visual(
        vid, x, y, w, h, "tableEx",
        {"Values": {"projections": [_col(e, c) for e, c in fields]}},
        z,
    )


def _column(vid: str, x: int, y: int, w: int, h: int, cat_e: str, cat_c: str, val_e: str, val_c: str, z: int = 0) -> dict:
    return _visual(
        vid, x, y, w, h, "clusteredColumnChart",
        {
            "Category": {"projections": [_col_active(cat_e, cat_c)]},
            "Y": {"projections": [_col(val_e, val_c)]},
        },
        z,
    )


def _write_page(pages_root: str, page_id: str, display_name: str, visuals: list[dict]) -> None:
    page_dir = os.path.join(pages_root, page_id)
    for spec in visuals:
        write_json(os.path.join(page_dir, "visuals", spec["name"], "visual.json"), spec)
    # Desktop discovers visuals from visuals/ subfolders — do not use visualContainers here.
    write_json(
        os.path.join(page_dir, "page.json"),
        {
            "$schema": PAGE_SCHEMA,
            "name": page_id,
            "displayName": display_name,
            "displayOption": "FitToPage",
            "height": 720,
            "width": 1280,
        },
    )


def build_report(report_id: str) -> None:
    """Create a complete 4-page dashboard with pre-built visuals."""
    pages_root = os.path.join(REPORT, "definition", "pages")
    if os.path.isdir(pages_root):
        shutil.rmtree(pages_root)

    # Hex page folder IDs (must match page.json name + pages.json pageOrder).
    page_defs = [
        ("e1000000000000000001", "Executive Summary"),
        ("e2000000000000000002", "Logistics & Shipping"),
        ("e3000000000000000003", "Inventory & Simulation"),
        ("e4000000000000000004", "Demand Forecasting"),
    ]

    write_json(
        os.path.join(REPORT, "definition.pbir"),
        {
            "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json",
            "version": "4.0",
            "datasetReference": {
                "byPath": {"path": "../SupplyChainAnalytics.SemanticModel"}
            },
        },
    )
    write_json(
        os.path.join(REPORT, ".platform"),
        {
            "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
            "metadata": {"type": "Report", "displayName": "SupplyChainAnalytics"},
            "config": {"version": "2.0", "logicalId": report_id},
        },
    )
    write_json(
        os.path.join(REPORT, "definition", "version.json"),
        {
            "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json",
            "version": "2.0.0",
        },
    )
    write_json(
        os.path.join(REPORT, "definition", "report.json"),
        {
            "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/1.2.0/schema.json",
            "layoutOptimization": "None",
        },
    )
    write_json(
        os.path.join(pages_root, "pages.json"),
        {
            "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.0.0/schema.json",
            "pageOrder": [p[0] for p in page_defs],
            "activePageName": page_defs[0][0],
        },
    )

    # Page 1 — Executive Summary (DAX measures + pre-aggregated charts)
    exec_visuals = [
        _card_meas("a1000000000000000001", 16, 16, 300, 110, "Fact_Orders", "Total Revenue", 1000),
        _card_meas("a1000000000000000002", 332, 16, 300, 110, "Fact_Orders", "Total Orders", 2000),
        _card_meas("a1000000000000000003", 648, 16, 300, 110, "Fact_Orders", "Total Profit", 3000),
        _card_meas("a1000000000000000004", 964, 16, 300, 110, "Fact_Orders", "On-Time Delivery %", 4000),
        _line("a1000000000000000005", 16, 140, 820, 280, "Monthly_Trend", "YearMonth", "Monthly_Trend", "Revenue", 5000),
        _donut("a1000000000000000006", 852, 140, 412, 280, "Category_KPIs", "Category", "Category_KPIs", "Revenue", 6000),
        _bar("a1000000000000000007", 16, 436, 1248, 260, "Region_KPIs", "Region", "Region_KPIs", "Revenue", 7000),
    ]
    _write_page(pages_root, page_defs[0][0], page_defs[0][1], exec_visuals)

    # Page 2 — Logistics
    log_visuals = [
        _bar("a2000000000000000001", 16, 16, 600, 320, "Shipping_KPIs", "Shipping_Mode", "Shipping_KPIs", "Revenue", 1000),
        _bar("a2000000000000000002", 632, 16, 632, 320, "Shipping_KPIs", "Shipping_Mode", "Shipping_KPIs", "On_Time_Pct", 2000),
        _bar("a2000000000000000003", 16, 352, 820, 340, "Country_KPIs", "Country", "Country_KPIs", "Revenue", 3000),
        _table("a2000000000000000004", 852, 352, 412, 340, [
            ("Olist_Delivery_KPIs", "State"),
            ("Olist_Delivery_KPIs", "Orders"),
            ("Olist_Delivery_KPIs", "Avg_Delivery_Days"),
        ], 4000),
    ]
    _write_page(pages_root, page_defs[1][0], page_defs[1][1], log_visuals)

    # Page 3 — Inventory
    inv_visuals = [
        _bar("a3000000000000000001", 16, 16, 600, 300, "ABC_Breakdown", "ABC_Class", "ABC_Breakdown", "Stockout_Reduction_Pct", 1000),
        _column("a3000000000000000002", 632, 16, 632, 300, "Inventory_Plan", "ABC", "Inventory_Plan", "annual_demand", 2000),
        _table("a3000000000000000003", 16, 332, 500, 360, [
            ("Simulation_KPIs", "Metric"),
            ("Simulation_KPIs", "Value"),
        ], 3000),
        _table("a3000000000000000004", 532, 332, 732, 360, [
            ("Inventory_Plan", "Product_Code"),
            ("Inventory_Plan", "annual_demand"),
            ("Inventory_Plan", "safety_stock"),
            ("Inventory_Plan", "ABC"),
        ], 4000),
    ]
    _write_page(pages_root, page_defs[2][0], page_defs[2][1], inv_visuals)

    # Page 4 — Demand
    dem_visuals = [
        _line("a4000000000000000001", 16, 16, 1248, 300, "Monthly_Trend", "YearMonth", "Monthly_Trend", "Units_Sold", 1000),
        _bar("a4000000000000000002", 16, 332, 600, 360, "Dim_Warehouse", "Warehouse", "Dim_Warehouse", "Total_Demand", 2000),
        _bar("a4000000000000000003", 632, 332, 632, 360, "Fact_Demand_Monthly", "Product_Category", "Fact_Demand_Monthly", "Total_Demand", 3000),
    ]
    _write_page(pages_root, page_defs[3][0], page_defs[3][1], dem_visuals)


def main() -> None:
    # Archive hand-written TMDL (invalid indentation prevented Desktop from opening)
    tmdl = os.path.join(SM, "definition")
    if os.path.isdir(tmdl):
        if os.path.isdir(TMDL_BACKUP):
            shutil.rmtree(TMDL_BACKUP)
        shutil.move(tmdl, TMDL_BACKUP)

    model_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())

    bim = {
        "name": "SupplyChainAnalytics",
        "compatibilityLevel": 1567,
        "model": {
            "culture": "en-US",
            "defaultPowerBIDataSourceVersion": "powerBI_V3",
            "annotations": [
                {"name": "__PBI_TimeIntelligenceEnabled", "value": "1"},
                {"name": "PBI_ProTooling", "value": '["DevMode"]'},
            ],
            "tables": [
                table(
                    "Fact_Orders",
                    "fact_orders.csv",
                    measures=[
                        {
                            "name": "Total Revenue",
                            "expression": "SUM(Fact_Orders[Revenue])",
                            "formatString": "$#,0",
                        },
                        {
                            "name": "Total Profit",
                            "expression": "SUM(Fact_Orders[Profit])",
                            "formatString": "$#,0",
                        },
                        {
                            "name": "Total Orders",
                            "expression": "COUNTROWS(Fact_Orders)",
                            "formatString": "#,0",
                        },
                        {
                            "name": "On-Time Delivery %",
                            "expression": "DIVIDE(SUM(Fact_Orders[On_Time]), COUNTROWS(Fact_Orders))",
                            "formatString": "0.0%",
                        },
                    ],
                ),
                table("Dim_Date", "dim_date.csv"),
                table("Dashboard_KPIs", "dashboard_kpis.csv"),
                table("KPI_Summary", "kpi_summary.csv"),
                table("Monthly_Trend", "monthly_trend.csv"),
                table("Category_KPIs", "category_kpis.csv"),
                table("Region_KPIs", "region_kpis.csv"),
                table("Country_KPIs", "country_kpis.csv"),
                table("Shipping_KPIs", "shipping_kpis.csv"),
                table("Supplier_Performance", "supplier_performance.csv"),
                table("Fact_Demand_Monthly", "fact_demand_monthly.csv"),
                table("Dim_Warehouse", "dim_warehouse.csv"),
                table("Inventory_Plan", "inventory_plan.csv"),
                table("Fact_Routes", "fact_routes.csv"),
                table("Olist_Delivery_KPIs", "olist_delivery_kpis.csv"),
                table("Simulation_KPIs", "simulation_kpis.csv"),
                table("ABC_Breakdown", "abc_breakdown.csv"),
                table("Simulation_Results", "simulation_results.csv"),
                table("Data_Source_Manifest", "data_source_manifest.csv"),
            ],
            "relationships": [
                {
                    "name": "Fact_Orders_Date",
                    "fromTable": "Fact_Orders",
                    "fromColumn": "Order_Date",
                    "toTable": "Dim_Date",
                    "toColumn": "Date",
                },
                {
                    "name": "Demand_Warehouse",
                    "fromTable": "Fact_Demand_Monthly",
                    "fromColumn": "Warehouse",
                    "toTable": "Dim_Warehouse",
                    "toColumn": "Warehouse",
                },
            ],
        },
    }

    write_json(os.path.join(SM, "model.bim"), bim)

    write_json(
        os.path.join(SM, "definition.pbism"),
        {"version": "1.0", "settings": {}},
    )

    write_json(
        os.path.join(SM, ".platform"),
        {
            "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
            "metadata": {
                "type": "SemanticModel",
                "displayName": "SupplyChainAnalytics",
            },
            "config": {"version": "2.0", "logicalId": model_id},
        },
    )

    build_report(report_id)

    # PBIP schema allows only "report" in artifacts; model is linked via definition.pbir.
    write_json(
        os.path.join(POWERBI, "SupplyChainAnalytics.pbip"),
        {
            "version": "1.0",
            "artifacts": [{"report": {"path": "SupplyChainAnalytics.Report"}}],
            "settings": {"enableAutoRecovery": True},
        },
    )

    print("Built Power BI Desktop project:")
    print(f"  {os.path.join(POWERBI, 'SupplyChainAnalytics.pbip')}")
    print(f"  model.bim -> {PBI_DATA.replace(chr(92)+chr(92), chr(92))}")
    print("\nOpen with:")
    print("  Double-click OPEN_POWERBI.bat in the project root")


if __name__ == "__main__":
    main()
