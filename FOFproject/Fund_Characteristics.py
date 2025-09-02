from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import math
import numpy as np
class Fund:
    def __init__(self, monthly_returns, performance_fee, management_fee):
        """
        monthly_returns: list of dicts, each with 'date' and 'value' keys
        performance_fee: float (e.g., 0.2 for 20%)
        management_fee: float (e.g., 0.01 for 1%)
        inception_date: str in 'YYYY-MM' format or None
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
        self.inception_date = min(entry['month'] for entry in self.monthly_returns) if self.monthly_returns else None
        self.latest_date = max(entry['month'] for entry in self.monthly_returns) if self.monthly_returns else None

    def __repr__(self):
        return (f"Fund(performance_fee={self.performance_fee}, "
                f"management_fee={self.management_fee}, "
                f"monthly_returns={len(self.monthly_returns)} entries)")

    def cumulative_return(self, start_month, end_month):
        """
        Calculates cumulative value from start_month to end_month (inclusive).

        Parameters
        ----------
        start_month : str
            Month string in 'YYYY-MM' or 'YYYY-M' format (e.g. '2024-7').
        end_month : str
            Month string in 'YYYY-MM' or 'YYYY-M' format (e.g. '2024-12').

        Returns
        -------
        float
            The cumulative return from start_month (exclusive) to end_month (inclusive).
        """

        # helper to parse flexible YYYY-M(M)
        def parse_month(mstr: str) -> datetime:
            return datetime.strptime(mstr, "%Y-%m")

        # normalize inputs
        start_dt = parse_month(start_month)
        end_dt = parse_month(end_month)

        value = 1.0
        for entry in self.monthly_returns:
            entry_dt = parse_month(entry["month"])
            if start_dt < entry_dt <= end_dt:
                value *= (1 + float(entry["value"]))

        return value
    
    def annualized_return(self, start_month, end_month):
        """
        Calculates annualized return from start_month to end_month (inclusive).
        start_month and end_month should be in 'YYYY-MM' format.
        """
        # Step 1: Compute cumulative return over the period
        cumulative = self.cumulative_return(start_month, end_month)

        # Step 2: Parse dates
        start_date = datetime.strptime(start_month, "%Y-%m")
        end_date = datetime.strptime(end_month, "%Y-%m")

        # Step 3: Calculate number of months in the period
        months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

        # Step 4: Annualize (compound return adjusted to yearly scale)
        annualized = cumulative ** (12 / months) - 1

        return annualized

    def volatility(self, start_month=None, end_month=None, ddof=1): 
        """Calculate the volatility of returns.

        Parameters
        ----------
        start_month: str, optional
            The month from which to start calculating volatility. The
            comparison is exclusive, meaning returns with a month strictly
            greater than ``start_month`` are included. If ``None`` (default),
            the calculation uses the beginning of the series.
        end_month: str, optional
            The final month (inclusive) to consider in the calculation. If
            ``None`` (default), the calculation uses the end of the series.

        Returns
        -------
        float
            The standard deviation of the selected series of monthly returns.
            If the range is empty, ``0.0`` is returned.
        """

        vals = []
        for entry in self.monthly_returns:
            m = entry["month"]
            if ((start_month is None or start_month <= m)
                and (end_month is None or m <= end_month)):
                try:
                    vals.append(float(entry["value"]))
                except (TypeError, ValueError):
                    # skip non-numeric values
                    continue

        if not vals:
            return 0.0

        s = pd.Series(vals, dtype="float64").dropna()
        if s.empty:
            return 0.0

        monthly_vol = float(s.std(ddof=ddof))
        return monthly_vol * math.sqrt(12.0)
    
    def sharpe_ratio(self, start_month=None, end_month=None, risk_free_rate=0.0):
        """Calculate the Sharpe ratio of returns.

        Parameters
        ----------
        start_month: str, optional
            The month from which to start calculating the Sharpe ratio. The
            comparison is exclusive, meaning returns with a month strictly
            greater than ``start_month`` are included. If ``None`` (default),
            the calculation uses the beginning of the series.
        end_month: str, optional
            The final month (inclusive) to consider in the calculation. If
            ``None`` (default), the calculation uses the end of the series.
        risk_free_rate: float, optional
            The annualized risk-free rate to use in the calculation. This is
            expressed as a decimal (e.g., ``0.03`` for 3%). Default is ``0.0``.

        Returns
        -------
        float
            The Sharpe ratio of the selected series of monthly returns.
            If the range is empty or volatility is zero, ``0.0`` is returned.
        """

        # Step 1: Calculate annualized return over the specified period
        ann_return = self.annualized_return(
            start_month=start_month,
            end_month=end_month,
        )

        # Step 2: Calculate volatility over the specified period
        vol = self.volatility(
            start_month=start_month,
            end_month=end_month,
        )

        # Step 3: Handle edge cases
        if vol == 0.0:
            return 0.0

        # Step 4: Compute Sharpe ratio
        sharpe = (ann_return - risk_free_rate) / vol

        return sharpe
    

    def sortino_ratio(self, start_month=None, end_month=None, risk_free_rate=0.0):
        """
        Calculate the annualized Sortino ratio from monthly returns.

        Parameters
        ----------
        start_month : str, optional
            Include months strictly greater than this (exclusive lower bound).
        end_month : str, optional
            Include months up to and including this (inclusive upper bound).
        risk_free_rate : float, default 0.0
            Annual risk-free rate (e.g., 0.02 for 2%).

        Returns
        -------
        float
            Annualized Sortino ratio. Returns 0.0 if no usable data.
        """

        # collect returns
        vals = [
            float(entry["value"])
            for entry in self.monthly_returns
            if ((start_month is None or start_month = entry["month"])
                and (end_month is None or entry["month"] <= end_month))
        ]
        if not vals:
            return 0.0

        s = np.array(vals)


        # convert annual risk-free rate to monthly
        monthly_rf = (1 + risk_free_rate) ** (1/12) - 1

        # excess returns
        excess_returns = s - monthly_rf

        # Downside returns (where returns < target)
        downside = np.minimum(0, s - monthly_rf)
        
        # Downside deviation (like std dev but only for negative returns)
        downside_deviation = np.sqrt(np.mean(downside**2))
        
        if downside_deviation == 0:
            return np.nan  # Avoid division by zero
        
        # Annualized Sortino ratio
        return np.mean(excess_returns) / downside_deviation * np.sqrt(12)
