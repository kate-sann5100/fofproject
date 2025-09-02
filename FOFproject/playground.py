import pandas as pd
import plotly.graph_objects as go
from Fund_Graphs import plot_cumulative_returns
from Fund_Characteristics import Fund

# Load return data from Excel
file_path = r"Z:\FOF Underlying Funds\RDGFF Fund Document\FOF Factsheet\RDGFF\2025\RETURN DATA.csv"
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
        monthly_returns=returns,
        performance_fee=0.2,
        management_fee=0.01,
    )

# Now you can access any fund by name:
TAIREN_fund = funds["HAO"]
# print(TAIREN_fund.monthly_returns)
# print(TAIREN_fund.cumulative_return(start_month="2025-06", end_month="2025-7"))
# print(TAIREN_fund.annualized_return(start_month=TAIREN_fund.inception_date, end_month=TAIREN_fund.latest_date))
# print(TAIREN_fund.sharpe_ratio(start_month=TAIREN_fund.inception_date, end_month=TAIREN_fund.latest_date, risk_free_rate=0.0))
# print(TAIREN_fund.volatility(start_month=TAIREN_fund.inception_date, end_month=TAIREN_fund.latest_date))
print(TAIREN_fund.sortino_ratio(start_month=TAIREN_fund.inception_date, end_month=TAIREN_fund.latest_date))


# # Plot cumulative returns
# fig = plot_cumulative_returns(
#     start_month="2019-01",
#     end_month="2025-6",
#     asset_columns=["LIM", "EUREKAHEDGE", "MSCI WORLD"],
#     df=df,
#     style="default"
# )
