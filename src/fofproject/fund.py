from datetime import datetime
from typing import List, Dict, Union
import plotly.graph_objects as go
import pandas as pd
import math
import numpy as np
from fofproject.utils import parse_month, list_of_dicts_to_df

class Fund:
    def __init__(self, name:str, monthly_returns: List[Dict], performance_fee: float, management_fee: float):
        """Initialize a Fund object.

        Args:
            name (str): Name of the fund
            monthly_returns (List[Dict]): List of monthly returns with 'date' and 'value' keys
            performance_fee (float): Performance fee as a decimal (e.g., 0.2 for 20%)
            management_fee (float): Management fee as a decimal (e.g., 0.01 for 1%)
        """
        processed_returns = []
        for entry in monthly_returns:
            raw_date = entry['date']
            # Try parsing date in 'DD/MM/YYYY' format
            dt = datetime.strptime(str(raw_date), '%d/%m/%Y')
            processed_returns.append({
                'datetime': dt, 
                'month': datetime(dt.year, dt.month, 1), 
                'value': entry['value']
            })
        self.name = name
        self.monthly_returns = processed_returns
        self.performance_fee = performance_fee
        self.management_fee = management_fee
        self.inception_date = self.compute_inception_date()
        self.latest_date = self.compute_latest_date()
        self.num_months = len(self.monthly_returns)
        self.total_cum_rtn = self.cumulative_return(self.inception_date, self.latest_date) if self.monthly_returns else None
        self.total_ann_rtn = self.annualized_return(self.inception_date, self.latest_date) if self.monthly_returns else None
        self.total_vol = self.volatility(self.inception_date, self.latest_date) if self.monthly_returns else None
        self.total_sharpe = self.sharpe_ratio(self.inception_date, self.latest_date) if self.monthly_returns else None
        self.total_sortino = self.sortino_ratio(self.inception_date, self.latest_date) if self.monthly_returns else None
        self.total_max_dd = self.max_drawdown(self.inception_date, self.latest_date) if self.monthly_returns else None
        self.total_pos_months = self.positive_months(self.inception_date, self.latest_date) if self.monthly_returns else None


    def compute_inception_date(self):
        return min(entry['month'] for entry in self.monthly_returns) if self.monthly_returns else None

    def compute_latest_date(self):
        return max(entry['month'] for entry in self.monthly_returns) if self.monthly_returns else None

    def __repr__(self):
        return (f"Fund(performance_fee={self.performance_fee}, "
                f"management_fee={self.management_fee}, "
                f"monthly_returns={len(self.monthly_returns)} entries)")

    def cumulative_return(self, start_month: Union[str, datetime], end_month: Union[str, datetime]) -> float:
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
        # convert str to datetime
        start_month = parse_month(start_month) if isinstance(start_month, str) else start_month
        end_month = parse_month(end_month) if isinstance(end_month, str) else end_month

        value = 1.0
        for entry in self.monthly_returns:
            if start_month <= entry["month"] <= end_month:
                value *= (1 + float(entry["value"]))
        return value - 1.0
    
    def annualized_return(self, start_month, end_month):
        """
        Calculates annualized return from start_month to end_month (inclusive).
        start_month and end_month should be in 'YYYY-MM' format.
        """
        # Step 1: Compute cumulative return over the period
        cumulative = self.cumulative_return(start_month, end_month)

        # Step 2: Parse dates
        start_date =  parse_month(start_month)
        end_date = parse_month(end_month)

        # Step 3: Calculate number of months in the period
        months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) +1

        # Step 4: Annualize (compound return adjusted to yearly scale)
        annualized = (1+cumulative) ** (12 / months) - 1

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
        start_month =  parse_month(start_month)
        start_ms = start_month.month + start_month.year * 12 if start_month else None
        end_month = parse_month(end_month)
        end_ms = end_month.month + end_month.year * 12 if end_month else None
        vals = []
        for entry in self.monthly_returns:
            m = entry["datetime"].month + entry["datetime"].year * 12
            if ((start_ms is None or start_ms <= m)
                and (end_ms is None or m <= end_ms)):
                try:
                    vals.append(float(entry["value"]))
                except (TypeError, ValueError):
                    # skip non-numeric values
                    continue

        if not vals:
            return 0.0

        s = pd.Series(vals, dtype="float64")
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

        start_month =  parse_month(start_month)
        end_month = parse_month(end_month)


        # collect returns
        vals = [
            float(entry["value"])
            for entry in self.monthly_returns
                if ((start_month is None or start_month <= entry["datetime"])
                    and (end_month is None or entry["datetime"] <= end_month))
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
        downside_deviation = np.sqrt((np.sum(downside**2)/(len(s)+1))) * np.sqrt(12)
        
        if downside_deviation == 0:
            return np.nan  # Avoid division by zero
        ann_return = np.prod(1 + excess_returns) ** (12 / len(excess_returns)) - 1
        sortino = (ann_return - risk_free_rate) / downside_deviation
        # Annualized Sortino ratio
        return sortino

    def max_drawdown(self, start_month=None, end_month=None):
        """
        Calculate the maximum drawdown from monthly returns.

        Parameters
        ----------
        start_month : str, optional
            Include months strictly greater than this (exclusive lower bound).
        end_month : str, optional
            Include months up to and including this (inclusive upper bound).

        Returns
        -------
        float
            Maximum drawdown as a decimal (e.g., 0.2 for 20%).
            Returns 0.0 if no usable data.
        """

        start_dt = parse_month(start_month)
        end_dt = parse_month(end_month)

        values = []
        cum_value = 1.0

        for entry in self.monthly_returns:
            entry_dt = parse_month(entry["month"])
            if start_dt <= entry_dt <= end_dt:
                cum_value *= (1 + float(entry["value"]))
                values.append(cum_value - 1.0)  # cumulative return up to this month

        cumulative = np.array(values)

        # Compute running maximum of cumulative returns
        running_max = np.maximum.accumulate(cumulative)

        # Compute drawdowns
        drawdowns = (running_max - cumulative) / (1 + running_max)

        # Maximum drawdown
        max_drawdown = np.max(drawdowns)

        return max_drawdown
    
    def positive_months(self, start_month=None, end_month=None):
        """
        Count the number of months with positive returns.

        Parameters
        ----------
        start_month : str, optional
            Include months strictly greater than this (exclusive lower bound).
        end_month : str, optional
            Include months up to and including this (inclusive upper bound).

        Returns
        -------
        int
            Number of months with positive returns.
        """

        start_dt = parse_month(start_month)
        end_dt = parse_month(end_month)

        count = 0
        total = 0
        for entry in self.monthly_returns:
            entry_dt = parse_month(entry["month"])
            if start_dt <= entry_dt <= end_dt and float(entry["value"]) > 0:
                count += 1
            total += 1

        return count/total if total > 0 else 0.0
    

    def return_in_positive_months(self, start_month=None, end_month=None):
        """
        Calculate cumulative return in months with positive returns.

        Parameters
        ----------
        start_month : str, optional
            Include months strictly greater than this (exclusive lower bound).
        end_month : str, optional
            Include months up to and including this (inclusive upper bound).

        Returns
        -------
        float
            Cumulative return in months with positive returns.
        """

        start_dt = parse_month(start_month)
        end_dt = parse_month(end_month)
        total_rtn = 0
        count = 0
        for entry in self.monthly_returns:
            entry_dt = parse_month(entry["month"])
            if start_dt <= entry_dt <= end_dt and float(entry["value"]) > 0:
                total_rtn += float(entry["value"]) 
                count += 1

        return total_rtn/count if count > 0 else 0.0
    
    def return_in_negative_months(self, start_month=None, end_month=None):
        """
        Calculate cumulative return in months with negative returns.

        Parameters
        ----------
        start_month : str, optional
            Include months strictly greater than this (exclusive lower bound).
        end_month : str, optional
            Include months up to and including this (inclusive upper bound).

        Returns
        -------
        float
            Cumulative return in months with negative returns.
        """

        start_dt = parse_month(start_month)
        end_dt = parse_month(end_month)
        total_rtn = 0
        count = 0
        for entry in self.monthly_returns:
            entry_dt = parse_month(entry["month"])
            if start_dt <= entry_dt <= end_dt and float(entry["value"]) < 0:
                total_rtn += float(entry["value"]) 
                count += 1

        return total_rtn/count if count > 0 else 0.0
    
    def correlation_to(self, benchmark_fund, start_month=None, end_month=None):
        """_summary_

        Args:
            benchmark_fund (Fund Class): A index that you want to compare/calculate the correlation with
            start_month (_type_): _description_
            end_month (_type_): _description_
        """
        # Example: list1 and list2 are your two lists
        df1 = list_of_dicts_to_df(self.monthly_returns, "value1")
        df2 = list_of_dicts_to_df(benchmark_fund.monthly_returns, "value2")

        # Merge on 'month' to align them
        merged = pd.merge(df1, df2, on="month", how="inner")
        merged.dropna
        if start_month == None and end_month == None:
            filtered = merged
        else: 
            filtered = merged[(merged["month"] >= parse_month(start_month)) & (df["month"] <= parse_month(end_month))]
        # Compute correlation
        corr = filtered["value1"].corr(filtered["value2"])
        return corr

    def beta_to(self, benchmark_fund, start_month=None, end_month=None):
        """
        Calculate the beta of this fund relative to a benchmark fund.

        Parameters
        ----------
        benchmark_fund : Fund
            The benchmark Fund object to compare against.
        start_month : str, optional
            Include months strictly greater than this (exclusive lower bound).
        end_month : str, optional
            Include months up to and including this (inclusive upper bound).

        Returns
        -------
        float
            Beta of this fund relative to the benchmark fund.
            Returns None if insufficient data.
        """

        covariance = self.total_vol * benchmark_fund.total_vol  * self.correlation_to(benchmark_fund,start_month=start_month,end_month=end_month)
        beta = covariance / (benchmark_fund.total_vol **2)
        return beta
