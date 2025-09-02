import pandas as pd
import plotly.graph_objects as go
from Fund_Graphs import plot_cumulative_returns


# Load return data from Excel
file_path = r"Z:\FOF Underlying Funds\RDGFF Fund Document\FOF Factsheet\RDGFF\2025\RETURN DATA.csv"
df = pd.read_csv(file_path)

from Fund_Characteristics import Fund
# Example usage:
# Extract date and Tairen columns (assuming 'date' and 'TAIREN' are the column names)
monthly_returns = []

for _, row in df.iterrows():
    date = row['date']  
    tairen_value = row['TAIREN'] 
    # Convert percentage string to float
    if pd.notnull(tairen_value):
        value = float(str(tairen_value).replace('%', '')) / 100
        monthly_returns.append({'date': date, 'value': value})

# Create Fund object for Tairen
tairen_fund = Fund(monthly_returns, performance_fee=0.2, management_fee=0.01)
result = tairen_fund.cumulative_return("2007-06", "2007-06")
print(result)

fig = plot_cumulative_returns(
    start_month="2020-01",
    end_month="2020-12",
    asset_columns=["RDGFF", "S&P 500", "MSCI WORLD"],
    df=df,
    style="default"
)

print(tairen_fund.monthly_returns[:5])
