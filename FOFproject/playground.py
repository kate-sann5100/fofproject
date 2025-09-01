import pandas as pd
import plotly.graph_objects as go

# Load return data from Excel
file_path = r"Z:\FOF Underlying Funds\RDGFF Fund Document\FOF Factsheet\RDGFF\2025\RETURN DATA.csv"
df = pd.read_csv(file_path)

from FundCharacteristics import Fund
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

print(tairen_fund.monthly_returns[:5])

# # Extract data for plotting
# years = df.iloc[:, 0].tolist()
# asia_values = df.iloc[:, 1].tolist()
# us_eu_values = df.iloc[:, 2].tolist()

# # Create figure
# fig = go.Figure()

# # Add Asia trace
# fig.add_trace(go.Scatter(x=years, y=asia_values, fill='tozeroy', name='Asia', line_color='gold'))

# # Add US/EU trace
# fig.add_trace(go.Scatter(x=years, y=us_eu_values, fill='tozeroy', name='US/EU', line_color='blue'))

# # Update layout
# fig.update_layout(
#     title='Asia vs US/EU Trend (2017-2026)',
#     xaxis_title='Year',
#     yaxis_title='Percentage',
#     yaxis_range=[0, 1],
# #     template='plotly_white'
# # )

# # Show plot
# fig.show()
