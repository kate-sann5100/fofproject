from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
class Fund:
    def __init__(self, monthly_returns, performance_fee, management_fee):
        """
        monthly_returns: list of dicts, each with 'date' and 'value' keys
        performance_fee: float (e.g., 0.2 for 20%)
        management_fee: float (e.g., 0.01 for 1%)
        """
        # Process date format to 'YYYY-MM' during initialization
        processed_returns = []
        for entry in monthly_returns:
            raw_date = entry['date']
            # Try parsing date in 'DD/MM/YYYY' format
            try:
                dt = datetime.strptime(str(raw_date), '%d/%m/%Y')
            except ValueError:
                # If already in 'YYYY-MM-DD', parse as such
                dt = datetime.strptime(str(raw_date), '%Y-%m-%d')
            entry_month = dt.strftime('%Y-%m')
            processed_returns.append({'month': entry_month, 'value': entry['value']})
        self.monthly_returns = processed_returns
        self.performance_fee = performance_fee
        self.management_fee = management_fee

    def __repr__(self):
        return (f"Fund(performance_fee={self.performance_fee}, "
                f"management_fee={self.management_fee}, "
                f"monthly_returns={len(self.monthly_returns)} entries)")

    def cumulative_return(self, start_month, end_month):
        """
        Calculates cumulative value from start_month to end_month (inclusive).
        start_month and end_month should be in 'YYYY-MM' format.
        """
        value = 1.0
        for entry in self.monthly_returns:
            entry_month = entry['month']
            if start_month < entry_month <= end_month:
                value *= (1 + entry['value'])
        return value


    def plot_cumulative_returns(self, start_month, end_month, asset_columns, df):
        """
        Plots cumulative returns for selected assets between start_month and end_month.
        fund: Fund object (for date processing)
        start_month, end_month: 'YYYY-MM' format
        asset_columns: list of column names in df to compare
        df: pandas DataFrame containing the raw data
        """
        months = []
        cumulative_returns = {asset: [] for asset in asset_columns}

        # Filter rows within the date range
        for _, row in df.iterrows():
            raw_date = row['date']
            # Parse date to 'YYYY-MM'
            try:
                dt = datetime.strptime(str(raw_date), '%d/%m/%Y')
            except ValueError:
                dt = datetime.strptime(str(raw_date), '%Y-%m-%d')
            entry_month = dt.strftime('%Y-%m')
            if start_month <= entry_month <= end_month:
                months.append(entry_month)
                for asset in asset_columns:
                    value = row[asset]
                    if pd.notnull(value):
                        ret = float(str(value).replace('%', '')) / 100
                    else:
                        ret = 0.0
                    # Calculate cumulative return
                    prev = cumulative_returns[asset][-1] if cumulative_returns[asset] else 1.0
                    cumulative_returns[asset].append(prev * (1 + ret))

        # Plot
        fig = go.Figure()
        for asset in asset_columns:
            fig.add_trace(go.Scatter(x=months, y=cumulative_returns[asset], mode='lines', name=asset))
        fig.update_layout(
            title='Cumulative Return Comparison',
            xaxis_title='Month',
            yaxis_title='Cumulative Return',
            template='plotly_white'
        )
        fig.show()

