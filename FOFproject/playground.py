import plotly.graph_objects as go

# Data for Asia and US/EU
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
asia_values = [100, 90, 80, 70, 60, 55, 50, 45, 40, 50]
us_eu_values = [0, 10, 20, 30, 40, 45, 50, 55, 60, 50]

# Create figure
fig = go.Figure()

# Add Asia trace
fig.add_trace(go.Scatter(x=years, y=asia_values, fill='tozeroy', name='Asia', line_color='gold'))

# Add US/EU trace
fig.add_trace(go.Scatter(x=years, y=us_eu_values, fill='tozeroy', name='US/EU', line_color='blue'))

# Update layout
fig.update_layout(
    title='Asia vs US/EU Trend (2017-2026)',
    xaxis_title='Year',
    yaxis_title='Percentage',
    yaxis_range=[0, 100],
    template='plotly_dark'
)

# Show plot
fig.show()
