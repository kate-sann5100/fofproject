import pandas as pd
import plotly.graph_objects as go
from src.fofproject.fund import Fund
from src.fofproject.plot import plot_cumulative_returns

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
keys = ['RDGFF', 'MSCI CHINA','MSCI WORLD','HAO','TAIREN']
funds_to_be_plot = {k: funds.get(k, None) for k in keys} # or a custom default

fig = plot_cumulative_returns(
    funds=funds_to_be_plot,
    title="Cumulative Returns",
    start_month="2020-1",
    end_month="2020-12",
    style="excel"
)
