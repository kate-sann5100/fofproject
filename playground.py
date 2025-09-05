import pandas as pd
import plotly.graph_objects as go
from src.fofproject.fund import Fund
from src.fofproject.plot import plot_cumulative_returns, plot_fund_correlation_heatmap
from src.fofproject.mvo import minimum_variance_analysis

# Load return data from Excel
file_path = r"RETURN DATA.csv"
df = pd.read_csv(file_path)

funds = {}
for col in df.columns:
    if col == "date":
        continue  # skip the date column


    returns = [
        {"date": d, "value": v}
        for d, v in zip(df["date"], df[col])
        if pd.notna(v)
    ]

    # Create a Fund instance
    funds[col] = Fund(
        name=col,
        monthly_returns=returns,
        performance_fee=0.2,
        management_fee=0.01,
    )

# print(funds['TAIREN'].total_cum_rtn)
# print(funds['TAIREN'].total_ann_rtn)   
# print(funds['TAIREN'].total_vol)
# # print(funds['RDGFF'].return_in_negative_months(start_month=funds['RDGFF'].inception_date,end_month=funds['RDGFF'].latest_date))
# keys = ['TAIREN', 'FOREST', 'LIM','3W GLOBAL','HAO','TIMEFOLIO','JH BIOTECH']
# funds_to_be_plot = {k: funds.get(k, None) for k in keys} # or a custom default
# # 基金回报历史走势


# fig = plot_cumulative_returns(
#     funds=funds_to_be_plot,
#     title="Cumulative Returns",
start_month="2020-1",
end_month="2025-12",
#     style="default",
#     language="en"
# )
for name, fund in funds.items():
    print(f"Running function for {name}")
    fund.annualized_return(start_month,)



# fig = funds["LEXINGTON"].plot_monthly_return_distribution()

# fig, corr_df, overlap_df = plot_fund_correlation_heatmap(funds, method="pearson", min_overlap=12)
# fig.show()

# All funds, long-only GMV, require at least 36 overlapping months



# fig, w, stats = minimum_variance_analysis(
#     funds=funds_to_be_plot,
#     long_only=True,
#     min_common_months=36,
#     title="GMV (Long-only, ≥36 common months)"
# )

# print(w.sort_values(ascending=False))
# print("Annualized vol:", f"{stats['ann_vol']:.2%}")
# print("Annualized return:", f"{stats['ann_ret']:.2%}")
# print("Number of months used:", stats['n_months'])

# print(funds['MSCI CHINA'].cumulative_return(start_month="2020-1", end_month="2020-12"))
# print(funds['MSCI WORLD'].cumulative_return(start_month="2020-1", end_month="2020-12"))
