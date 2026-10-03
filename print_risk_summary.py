import json

with open('reports/forecast_risk_report.json', encoding='utf-8') as f:
    r = json.load(f)

print('=' * 50)
print('  OVERSTOCK & STOCKOUT RISK SUMMARY')
print('  Source: reports/forecast_risk_report.json')
print('=' * 50)

for m in r['models']:
    ov = m['overstock']
    so = m['stockout']
    fa = m['forecast_accuracy']
    print()
    print(f"Model : {m['model']}")
    print(f"  Date range        : {m['date_range']['start']}  to  {m['date_range']['end']}")
    print(f"  Periods evaluated : {m['periods_evaluated']}")
    print()
    print(f"  Overstock Risk    : {ov['avg_overstock_pct']:>7.2f} %  [{ov['risk_level']}]")
    print(f"    Max single week : {ov['max_overstock_pct']:>7.2f} %")
    print(f"    Weeks affected  : {ov['periods_overstocked']} of {m['periods_evaluated']}  ({ov['overstock_frequency_pct']} % of all periods)")
    print()
    print(f"  Stockout Risk     : {so['avg_stockout_pct']:>7.2f} %  [{so['risk_level']}]")
    print(f"    Max single week : {so['max_stockout_pct']:>7.2f} %")
    print(f"    Weeks affected  : {so['periods_stockout']} of {m['periods_evaluated']}  ({so['stockout_frequency_pct']} % of all periods)")
    print()
    print(f"  MAPE              : {fa['MAPE_pct']:>7.2f} %")
    print(f"  Bias direction    : {fa['bias']}")
    print('-' * 50)

e = r['ensemble']
print()
print(f"ENSEMBLE  ({e['periods_evaluated']} matched periods)")
print(f"  Avg Overstock Risk : {e['avg_overstock_pct']:>7.2f} %")
print(f"  Avg Stockout  Risk : {e['avg_stockout_pct']:>7.2f} %")
print()
print(f"  Note: {e['note']}")
print('=' * 50)
