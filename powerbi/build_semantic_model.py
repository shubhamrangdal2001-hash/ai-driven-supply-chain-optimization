"""
Generate TMDL semantic model files for SupplyChainAnalytics.pbip.
Run after powerbi_data_prep.py. Uses tab indentation required by TMDL.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBI_DATA = os.path.join(ROOT, "reports", "powerbi").replace("/", "\\")
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "SupplyChainAnalytics.SemanticModel",
    "definition",
)
TABLES_DIR = os.path.join(MODEL_DIR, "tables")


def m_csv(filename: str, transforms: str = "") -> str:
    base = f"""let
\t    Source = Csv.Document(
\t        File.Contents(PBI_DataPath & "\\\\{filename}"),
\t        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]
\t    ),
\t    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true])"""
    if transforms:
        return f"{base},\n{transforms}\nin\n\t#" + transforms.split("\n")[0].strip().split("=")[0].strip()
    return f"{base}\nin\n\t#" + "Promoted Headers"


def m_csv_typed(filename: str, type_steps: list[str]) -> str:
    lines = [
        "let",
        "\t    Source = Csv.Document(",
        f'\t        File.Contents(PBI_DataPath & "\\\\{filename}"),',
        '\t        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]',
        "\t    ),",
        '\t    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),',
    ]
    for step in type_steps:
        lines.append(f"\t    {step},")
    last = type_steps[-1].split("=")[0].strip() if type_steps else '"Promoted Headers"'
    lines.append("in")
    lines.append(f"\t    {last}")
    return "\n".join(lines)


def write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def main() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)

    write(
        os.path.join(MODEL_DIR, "database.tmdl"),
        f"""database SupplyChainAnalytics
\tcompatibilityLevel: 1567
""",
    )

    write(
        os.path.join(MODEL_DIR, "model.tmdl"),
        """model Model
\tculture: en-US
\tdefaultPowerBIDataSourceVersion: powerBI_V3
\tannotation __PBI_TimeIntelligenceEnabled = 1
\tannotation PBI_ProTooling = ["DevMode"]

\tref table Fact_Orders
\tref table Dim_Date
\tref table KPI_Summary
\tref table Monthly_Trend
\tref table Category_KPIs
\tref table Region_KPIs
\tref table Country_KPIs
\tref table Shipping_KPIs
\tref table Supplier_Performance
\tref table Fact_Demand_Monthly
\tref table Dim_Warehouse
\tref table Inventory_Plan
\tref table Fact_Routes
\tref table Olist_Delivery_KPIs
\tref table Simulation_KPIs
\tref table ABC_Breakdown
\tref table Simulation_Results
\tref table Data_Source_Manifest

\tref cultureInfo en-US
""",
    )

    write(
        os.path.join(MODEL_DIR, "expressions.tmdl"),
        f"""expression PBI_DataPath = "{PBI_DATA}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]
""",
    )

    write(
        os.path.join(MODEL_DIR, "relationships.tmdl"),
        """relationship Fact_Orders_Date
\tfromColumn: Fact_Orders.Order_Date
\ttoColumn: Dim_Date.Date

relationship Demand_Warehouse
\tfromColumn: Fact_Demand_Monthly.Warehouse
\ttoColumn: Dim_Warehouse.Warehouse
""",
    )

    write(
        os.path.join(MODEL_DIR, "cultures", "en-US.tmdl"),
        """cultureInfo en-US
\tlinguisticMetadata =
\t\t\t{
\t\t\t  "Version": "1.0.0",
\t\t\t  "Language": "en-US"
\t\t\t}
\t\tcontentType: json
""",
    )

  # Fact_Orders
    fact_m = m_csv_typed(
        "fact_orders.csv",
        [
            '#"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers", {{"Order_Date", type date}, {"Order_Year", Int64.Type}, {"Order_Month", Int64.Type}, {"Revenue", type number}, {"Profit", type number}, {"Quantity", Int64.Type}, {"On_Time", Int64.Type}})',
        ],
    )
    write(
        os.path.join(TABLES_DIR, "Fact_Orders.tmdl"),
        f"""table Fact_Orders
\tlineageTag: fact-orders-001

\tmeasure 'Total Revenue' = SUM(Fact_Orders[Revenue])
\t\tformatString: "$#,0"
\t\tdisplayFolder: Orders

\tmeasure 'Total Profit' = SUM(Fact_Orders[Profit])
\t\tformatString: "$#,0"
\t\tdisplayFolder: Orders

\tmeasure 'Total Orders' = COUNTROWS(Fact_Orders)
\t\tformatString: "#,0"
\t\tdisplayFolder: Orders

\tmeasure 'On-Time Delivery %' = DIVIDE(SUM(Fact_Orders[On_Time]), COUNTROWS(Fact_Orders))
\t\tformatString: 0.0%
\t\tdisplayFolder: Logistics

\tmeasure 'Avg Order Value' = DIVIDE([Total Revenue], [Total Orders])
\t\tformatString: "$#,0.00"
\t\tdisplayFolder: Orders

\tmeasure 'Profit Margin %' = DIVIDE([Total Profit], [Total Revenue])
\t\tformatString: 0.0%
\t\tdisplayFolder: Orders

\tcolumn Order_Date
\t\tdataType: dateTime
\t\tformatString: Short Date
\t\tsourceColumn: Order_Date

\tcolumn Order_Year
\t\tdataType: int64
\t\tformatString: 0
\t\tsourceColumn: Order_Year
\t\tsummarizeBy: none

\tcolumn Order_Month
\t\tdataType: int64
\t\tformatString: 0
\t\tsourceColumn: Order_Month
\t\tsummarizeBy: none

\tcolumn Category
\t\tdataType: string
\t\tsourceColumn: Category

\tcolumn Market
\t\tdataType: string
\t\tsourceColumn: Market

\tcolumn Region
\t\tdataType: string
\t\tsourceColumn: Region

\tcolumn Country
\t\tdataType: string
\t\tsourceColumn: Country
\t\tdataCategory: Country

\tcolumn Shipping_Mode
\t\tdataType: string
\t\tsourceColumn: Shipping_Mode

\tcolumn Revenue
\t\tdataType: double
\t\tformatString: "$#,0.00"
\t\tsourceColumn: Revenue

\tcolumn Profit
\t\tdataType: double
\t\tformatString: "$#,0.00"
\t\tsourceColumn: Profit

\tcolumn Quantity
\t\tdataType: int64
\t\tformatString: "#,0"
\t\tsourceColumn: Quantity

\tcolumn On_Time
\t\tdataType: int64
\t\tformatString: 0
\t\tsourceColumn: On_Time
\t\tsummarizeBy: none

\tcolumn Delay_Days
\t\tdataType: double
\t\tformatString: "0.00"
\t\tsourceColumn: Delay_Days

\tpartition Fact_Orders = m
\t\tmode: import
\t\tsource =
\t\t\t\t{fact_m.replace(chr(10), chr(10) + chr(9) + chr(9))}
""",
    )

    dim_date_m = m_csv_typed(
        "dim_date.csv",
        [
            '#"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers", {{"Date", type date}, {"DateKey", Int64.Type}})',
        ],
    )
    write(
        os.path.join(TABLES_DIR, "Dim_Date.tmdl"),
        f"""table Dim_Date
\tlineageTag: dim-date-001

\tcolumn Date
\t\tdataType: dateTime
\t\tformatString: Short Date
\t\tsourceColumn: Date
\t\tisKey

\tcolumn DateKey
\t\tdataType: int64
\t\tformatString: 0
\t\tsourceColumn: DateKey
\t\tisHidden
\t\tsummarizeBy: none

\tcolumn Year
\t\tdataType: int64
\t\tformatString: 0
\t\tsourceColumn: Year
\t\tsummarizeBy: none

\tcolumn Month_Name
\t\tdataType: string
\t\tsourceColumn: Month_Name

\tcolumn YearMonth
\t\tdataType: string
\t\tsourceColumn: YearMonth

\tpartition Dim_Date = m
\t\tmode: import
\t\tsource =
\t\t\t\t{dim_date_m.replace(chr(10), chr(10) + chr(9) + chr(9))}
""",
    )

    simple_tables = [
        ("KPI_Summary", "kpi_summary.csv", "KPI_Summary"),
        ("Monthly_Trend", "monthly_trend.csv", "Monthly_Trend"),
        ("Category_KPIs", "category_kpis.csv", "Category_KPIs"),
        ("Region_KPIs", "region_kpis.csv", "Region_KPIs"),
        ("Country_KPIs", "country_kpis.csv", "Country_KPIs"),
        ("Shipping_KPIs", "shipping_kpis.csv", "Shipping_KPIs"),
        ("Supplier_Performance", "supplier_performance.csv", "Supplier_Performance"),
        ("Fact_Demand_Monthly", "fact_demand_monthly.csv", "Fact_Demand_Monthly"),
        ("Dim_Warehouse", "dim_warehouse.csv", "Dim_Warehouse"),
        ("Inventory_Plan", "inventory_plan.csv", "Inventory_Plan"),
        ("Fact_Routes", "fact_routes.csv", "Fact_Routes"),
        ("Olist_Delivery_KPIs", "olist_delivery_kpis.csv", "Olist_Delivery_KPIs"),
        ("Simulation_KPIs", "simulation_kpis.csv", "Simulation_KPIs"),
        ("ABC_Breakdown", "abc_breakdown.csv", "ABC_Breakdown"),
        ("Simulation_Results", "simulation_results.csv", "Simulation_Results"),
        ("Data_Source_Manifest", "data_source_manifest.csv", "Data_Source_Manifest"),
    ]

    for table_name, filename, partition_name in simple_tables:
        src = m_csv(filename)
        write(
            os.path.join(TABLES_DIR, f"{table_name}.tmdl"),
            f"""table {table_name}
\tlineageTag: {table_name.lower()}-001

\tpartition {partition_name} = m
\t\tmode: import
\t\tsource =
\t\t\t\t{src.replace(chr(10), chr(10) + chr(9) + chr(9))}
""",
        )

    # Inventory measures
    inv_path = os.path.join(TABLES_DIR, "Inventory_Plan.tmdl")
    with open(inv_path, "r", encoding="utf-8") as f:
        inv_content = f.read()
    inv_content = inv_content.replace(
        "table Inventory_Plan",
        """table Inventory_Plan

\tmeasure 'SKU Count' = COUNTROWS(Inventory_Plan)
\t\tformatString: "#,0"
\t\tdisplayFolder: Inventory

\tmeasure 'Avg Safety Stock' = AVERAGE(Inventory_Plan[safety_stock])
\t\tformatString: "#,0"
\t\tdisplayFolder: Inventory""",
        1,
    )
    write(inv_path, inv_content)

    write(
        os.path.join(os.path.dirname(MODEL_DIR), "definition.pbism"),
        """{
  "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json",
  "version": "4.0",
  "settings": {}
}
""",
    )

    write(
        os.path.join(
            os.path.dirname(os.path.dirname(MODEL_DIR)),
            "SupplyChainAnalytics.pbip",
        ),
        """{
  "version": "1.0",
  "artifacts": [
    {
      "report": {
        "path": "SupplyChainAnalytics.Report"
      }
    }
  ],
  "settings": {
    "enableAutoRecovery": true
  }
}
""",
    )

    report_dir = os.path.join(
        os.path.dirname(os.path.dirname(MODEL_DIR)),
        "SupplyChainAnalytics.Report",
    )
    write(
        os.path.join(report_dir, "definition.pbir"),
        """{
  "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json",
  "version": "4.0",
  "datasetReference": {
    "byPath": {
      "path": "../SupplyChainAnalytics.SemanticModel"
    }
  }
}
""",
    )

    write(
        os.path.join(report_dir, "definition", "version.json"),
        """{
  "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json",
  "version": "2.0.0"
}
""",
    )

    write(
        os.path.join(report_dir, "definition", "report.json"),
        """{
  "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/1.2.0/schema.json",
  "themeCollection": {
    "baseTheme": {
      "name": "CY24SU10",
      "reportVersionAtImport": "5.55",
      "type": "SharedResources"
    }
  },
  "layoutOptimization": "None",
  "resourcePackages": [
    {
      "name": "SharedResources",
      "type": "SharedResources",
      "items": [
        {
          "name": "CY24SU10",
          "path": "BaseThemes/CY24SU10.json",
          "type": "BaseTheme"
        }
      ]
    }
  ]
}
""",
    )

    # Executive summary page
    write(
        os.path.join(report_dir, "definition", "pages", "pages.json"),
        """{
  "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.0.0/schema.json",
  "pageOrder": [
    "executive_summary",
    "logistics",
    "inventory",
    "demand"
  ],
  "activePageName": "executive_summary"
}
""",
    )

    for page_id, display_name in [
        ("executive_summary", "Executive Summary"),
        ("logistics", "Logistics & Shipping"),
        ("inventory", "Inventory & Simulation"),
        ("demand", "Demand Forecasting"),
    ]:
        write(
            os.path.join(report_dir, "definition", "pages", page_id, "page.json"),
            f"""{{
  "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/1.3.0/schema.json",
  "name": "{page_id}",
  "displayName": "{display_name}",
  "displayOption": "FitToPage",
  "height": 720,
  "width": 1280
}}
""",
        )

    print(f"TMDL semantic model generated at: {MODEL_DIR}")
    print("Open SupplyChainAnalytics.pbip in Power BI Desktop")


if __name__ == "__main__":
    main()
