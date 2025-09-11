import pandas as pd
import plotly.graph_objects as go
from src.fofproject.fund import Fund, input_monthly_returns, subset_of_funds
from src.fofproject.plot import plot_cumulative_returns, plot_fund_correlation_heatmap
from src.fofproject.mvo import minimum_variance_analysis

# Initialize and load data
funds = input_monthly_returns(r"RETURN DATA.csv", performance_fee=0.2, management_fee=0.01)
# funds_to_be_plot = subset_of_funds(funds, ['RDGFF', 'EUREKAHEDGE','MSCI CHINA'])
funds_to_be_plot = subset_of_funds(funds, ['RDGFF', 'EUREKAHEDGE','MSCI CHINA', 'HAO','TAIREN', 'LEXINGTON', 'LIM','FOREST'])
start_month = "2017-1"
end_month = "2070-8"

# for name, fund in funds_to_be_plot.items():
#     print(
#         f"Fund: {name}, return during the period {funds[name].annualized_return('2020-05', end_month):.2f}, "
#         f"volatility during the period {funds[name].volatility('2020-05', end_month):.2f}, "
#         f"sharpe during the period {funds[name].sharpe_ratio('2020-05', end_month):.2f}"
#     )

print(funds['RDGFF'].beta_to(funds['MSCI CHINA'], '2000-1', end_month))
print(funds['RDGFF'].correlation_to(funds['MSCI CHINA'], '2000-1', end_month))

# Correlation heatmap
# fig, corr_df, overlap_df = plot_fund_correlation_heatmap(funds_to_be_plot, method="pearson", min_overlap=12)
# fig.show()

# # 基金回报历史走势

# # MVO analysis
# fig, w, stats = minimum_variance_analysis(
#     funds=funds_to_be_plot,
#     long_only=True,
#     min_common_months=36,
#     title="GMV (Long-only, ≥36 common months)"
# )

# # Cumulative returns
# fig = plot_cumulative_returns(
#     funds=funds_to_be_plot,
#     title=title,
#     start_month=start_month,
#     end_month=end_month,
#     style="modern_dark",
#     language="en",
#     blur=True
# )
# for name, fund in funds.items():
#     print(f"Running function for {name}")
#     fund.annualized_return(start_month, end_month)


# l = "plot_monthly_return_distribution()"
# fig = eval(f"funds['LEXINGTON'].{l}")


