import pandas as pd
import plotly.graph_objects as go
from src.fofproject.fund import Fund
from src.fofproject.plot import plot_cumulative_returns, plot_fund_correlation_heatmap

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
# print(funds['RDGFF'].return_in_negative_months(start_month=funds['RDGFF'].inception_date,end_month=funds['RDGFF'].latest_date))
keys = ['RDGFF', 'MSCI CHINA','MSCI WORLD']
funds_to_be_plot = {k: funds.get(k, None) for k in keys} # or a custom default
# 基金回报历史走势


# fig = plot_cumulative_returns(
#     funds=funds_to_be_plot,
#     title="Cumulative Returns",
#     start_month="2020-1",
#     end_month="2020-12",
#     style="default",
#     language="en"
# )

# fig = funds["LEXINGTON"].plot_monthly_return_distribution()

fig, corr_df, overlap_df = plot_fund_correlation_heatmap(funds, method="pearson", min_overlap=12)
fig.show()

# print(funds['MSCI CHINA'].cumulative_return(start_month="2020-1", end_month="2020-12"))
# print(funds['MSCI WORLD'].cumulative_return(start_month="2020-1", end_month="2020-12"))
