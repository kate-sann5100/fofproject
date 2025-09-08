import pandas as pd
import plotly.graph_objects as go
from src.fofproject.fund import Fund, input_monthly_returns, subset_of_funds
from src.fofproject.plot import plot_cumulative_returns, plot_fund_correlation_heatmap
from src.fofproject.mvo import minimum_variance_analysis

# Initialize and load data
funds = input_monthly_returns(r"RETURN DATA.csv", performance_fee=0.2, management_fee=0.01)
funds_to_be_plot = subset_of_funds(funds, ['RDGFF', 'FOREST', 'LIM','3W GLOBAL','HAO','TIMEFOLIO','JH BIOTECH'])
start_month = "2020-1"
end_month = "2025-6"


# print(funds['RDGFF'].annualized_return(funds['RDGFF'].inception_date, end_month))
# print(funds['RDGFF'].sortino_ratio(funds['RDGFF'].inception_date, end_month))

# 基金回报历史走势

# MVO analysis
# fig, w, stats = minimum_variance_analysis(
#     funds=funds_to_be_plot,
#     long_only=True,
#     min_common_months=36,
#     title="GMV (Long-only, ≥36 common months)"
# )

# Cumulative returns
fig = plot_cumulative_returns(
    funds=funds_to_be_plot,
    title="Cumulative Returns",
    start_month=start_month,
    end_month=end_month,
    style="default",
    language="en"
)
# for name, fund in funds.items():
#     print(f"Running function for {name}")
#     fund.annualized_return(start_month, end_month)


# l = "plot_monthly_return_distribution()"
# fig = eval(f"funds['LEXINGTON'].{l}")

# Correlation heatmap
# fig, corr_df, overlap_df = plot_fund_correlation_heatmap(funds, method="pearson", min_overlap=12)
# fig.show()
