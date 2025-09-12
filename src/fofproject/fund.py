from datetime import datetime
from typing import List, Dict, Union
import plotly.graph_objects as go
import pandas as pd
import math
import numpy as np
from fofproject.utils import parse_month, list_of_dicts_to_df, hex_to_rgba

def input_monthly_returns(file_path, performance_fee = 0.2, management_fee = 0.01):
    """Read monthly returns from a CSV file and create Fund instances."""
    # Read CSV file
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
            performance_fee=performance_fee,
            management_fee=management_fee,
        )
    return funds

def subset_of_funds(funds,keys=None):
    """funds: dict of Fund instances;
       keys: list of fund names to extract"""
    default = ['RDGFF', 'MSCI CHINA', 'MSCI GLOBAL']
    if keys is None:
        keys = default
    funds_to_be_plot = {k: funds.get(k, None) for k in keys} # or a custom default
    return funds_to_be_plot

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

    def get_monthly_return(self, year: int, month: int):
        target = datetime(year, month, 1)
        for entry in self.monthly_returns:
            if entry['month'] == target:
                return entry['value']
        return None
    
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
        ann_return = self.annualized_return(start_month, end_month)
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
    
    def join_two_funds(self, benchmark_fund, start_month=None, end_month=None):

        fund1 = self.monthly_returns   # market returns
        fund2 = benchmark_fund.monthly_returns   # stock returns

        req_start = parse_month(start_month)
        req_end   = parse_month(end_month)

        # gather available months
        f1_months = [e['month'] for e in fund1]
        f2_months = [e['month'] for e in fund2]

        earliest_common_start = max(min(f1_months), min(f2_months))
        latest_common_end = min(max(f1_months), max(f2_months))
        # final clamped range (also respect the user’s requested window)
        adj_start = max(req_start, earliest_common_start)
        adj_end   = min(req_end,   latest_common_end)

        if adj_start > adj_end:
            return [], [], (adj_start, adj_end)  # or raise, depending on your needs

        fund1_values = [
            entry['value']
            for entry in fund1
            if adj_start <= entry['month'] <= adj_end
        ]

        fund2_values = [
            entry['value']
            for entry in fund2
            if adj_start <= entry['month'] <= adj_end
        ]
        return fund1_values, fund2_values
    
    def correlation_to(self, benchmark_fund, start_month=None, end_month=None):
        """_summary_

        Args:
            benchmark_fund (Fund Class): A index that you want to compare/calculate the correlation with
            start_month (_type_): _description_
            end_month (_type_): _description_
        """
        # Example: list1 and list2 are your two lists
        fund1_values, fund2_values = self.join_two_funds(benchmark_fund=benchmark_fund, start_month=start_month, end_month=end_month)
        arr1 = np.array(fund1_values)
        arr2 = np.array(fund2_values)

        # correlation matrix
        corr_matrix = np.corrcoef(arr1, arr2)

        # extract the correlation coefficient
        corr = corr_matrix[0, 1]
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
               # Example: assume you already have two lists of dicts
        fund1_values, fund2_values = self.join_two_funds(benchmark_fund=benchmark_fund, start_month=start_month, end_month=end_month)

        # Combine into 2D numpy array (rows = months, columns = funds)
        np_array = np.column_stack([fund1_values, fund2_values])

        # Now you can access
        m = np_array[:, 1]   # fund 1 (market returns)
        s = np_array[:, 0]   # fund 2 (stock returns)
        covariance = np.cov(s,m) # Calculate covariance between stock and market
        beta = covariance[0,1]/covariance[1,1]

        return beta

    def plot_monthly_return_distribution(
        self,
        *,
        start_month: str | None = None,   # "YYYY-MM"
        end_month: str | None = None,     # "YYYY-MM"
        bins: int = 24,
        value_key: str = "value",         # key in each monthly_returns entry
        show_stats_lines: bool = True
    ):
        """
        Plot a histogram (bar chart) of this fund's historical monthly returns.
        Style is defined directly inside the function (no external STYLE_DICT).
        Adds a smoothed KDE curve for a softer distribution edge.
        """
        import numpy as np
        import plotly.graph_objects as go
        from math import sqrt, pi, exp

        # --- small helper (in case it's not already defined) ---
        def hex_to_rgba(hx: str, alpha: float = 1.0) -> str:
            hx = hx.lstrip("#")
            r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        # ----- style + palette (inlined here) -----
        layout_config = {
            "font": dict(family="Montserrat, Roboto", size=14, color="#53565A"),
            "margin": dict(l=54, r=42, t=84, b=72),
            "grid_color": "#e9e9ea"
        }
        color = "#C1AE94"   # keep your requested color
        fund_name = self.name or "Fund"

        # ----- clamp date window to available history -----
        sm = parse_month(start_month) if start_month else self.inception_date
        em = parse_month(end_month) if end_month else self.latest_date

        if not self.monthly_returns:
            raise ValueError("No monthly_returns available on this fund.")

        months_all = [e.get("month") for e in self.monthly_returns if "month" in e]
        first_m, last_m = months_all[0], months_all[-1]
        sm = sm or first_m
        em = em or last_m
        if sm > em:
            raise ValueError("start_month must be <= end_month")

        # ----- extract values -----
        vals = []
        for e in self.monthly_returns:
            m = e.get("month")
            if m is None or not (sm <= m <= em):
                continue
            if value_key in e:
                vals.append(float(e[value_key]))
            else:
                for k_guess in ("return", "ret", "monthly_return", "value"):
                    if k_guess in e:
                        vals.append(float(e[k_guess]))
                        break

        if not vals:
            raise ValueError(f"No monthly return values found for {fund_name} in the requested window.")

        vals = np.array(vals, dtype=float)
        n = len(vals)
        mean_r = float(np.mean(vals))
        p5, p50, p95 = (float(x) for x in np.percentile(vals, [5, 50, 95]))

        # ----- histogram bin width (for KDE scaling into % per bin) -----
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        # guard against zero-width
        rng = vmax - vmin if vmax > vmin else max(abs(vmax), 1e-6)
        bin_width = rng / max(bins, 1)

        # ----- simple Gaussian KDE (no scipy) -----
        # Scott's rule-of-thumb bandwidth
        std = float(np.std(vals)) if n > 1 else 1e-6
        h = 1.06 * std * (n ** (-1/5)) if n > 1 and std > 0 else (rng / 20.0 or 1e-3)

        x_grid = np.linspace(vmin - bin_width, vmax + bin_width, 400)

        def gaussian_kernel(u):
            return (1.0 / sqrt(2 * pi)) * np.exp(-0.5 * u * u)

        if n > 0:
            # density f(x), integrates to 1
            diffs = (x_grid[:, None] - vals[None, :]) / h
            dens = np.mean(gaussian_kernel(diffs), axis=1) / h
            # Scale to approximate histogram '% of months per bin' at each x:
            kde_percent = dens * bin_width * 100.0
        else:
            kde_percent = np.zeros_like(x_grid)

        # ----- figure -----
        fig = go.Figure()

        # Histogram with softer edges: semi-transparent fill + lighter outline
        fig.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=bins,
                histnorm="percent",
                marker=dict(
                    color=hex_to_rgba(color, 0.55),
                    line=dict(color=hex_to_rgba(color, 0.85), width=0.8)
                ),
                opacity=0.95,
                hovertemplate=(
                    "<b>%{x.start:.2%} – %{x.end:.2%}</b>"
                    "<br>%{y:.1f}% of months"
                    "<extra></extra>"
                ),
                name=fund_name,
                showlegend=False
            )
        )

        # Smooth KDE curve on top for a softer distribution "edge"
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=kde_percent,
                mode="lines",
                line=dict(color=color, width=3),
                name="KDE",
                hovertemplate="<b>%{x:.2%}</b><br>%{y:.2f}% (smooth)<extra></extra>",
                showlegend=False
            )
        )

        # ----- reference lines -----
        if show_stats_lines:
            # 0% vertical line
            fig.add_shape(
                type="line", x0=0, x1=0, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="#2F2F2F", width=1, dash="dash")
            )
            fig.add_annotation(
                x=0, y=1.02, xref="x", yref="paper",
                text="0%", showarrow=False, font=dict(size=12, color="#666")
            )
            # mean line
            fig.add_shape(
                type="line", x0=mean_r, x1=mean_r, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color=color, width=2)
            )
            fig.add_annotation(
                x=mean_r, y=1.02, xref="x", yref="paper",
                text=f"Mean {mean_r:.2%}", showarrow=False,
                font=dict(size=12, color=color)
            )

        # ----- layout tweaks for a cleaner, smoother feel -----
        fig.update_layout(
            title=dict(
                text=f"<b>{fund_name} — Monthly Return Distribution</b>",
                font=dict(size=26),
                x=0.5, xanchor="center", y=0.97, yanchor="middle"
            ),
            template="plotly_white",
            font=layout_config["font"],
            margin=layout_config["margin"],
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="x unified",
            hoverlabel=dict(
                font=dict(family=layout_config["font"]["family"], size=13, color="#333"),
                bgcolor="white",
                bordercolor=hex_to_rgba(color, 0.6)
            ),
            xaxis=dict(
                title="Monthly Return",
                tickformat="+.0%",
                showgrid=True,
                gridcolor=layout_config["grid_color"],
                zeroline=False,
                ticks="outside",
                ticklen=6,
                tickcolor="#d7d7d9"
            ),
            yaxis=dict(
                title="Months (%)",
                ticksuffix="%",
                showgrid=True,
                gridcolor=layout_config["grid_color"],
                zeroline=False,
                ticks="outside",
                ticklen=6,
                tickcolor="#d7d7d9"
            ),
            bargap=0.35
        )

        # small stats box (subtle, right-top)
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            align="right", showarrow=False,
            text=(
                f"<span style='color:{color};'><b>{fund_name}</b></span><br>"
                f"n = {n}<br>"
                f"Mean = {mean_r:.2%}<br>"
                f"Median = {p50:.2%}<br>"
                f"P5 / P95 = {p5:.2%} / {p95:.2%}"
            ),
            bgcolor="#F6F6F7",
            bordercolor=hex_to_rgba(color, 0.9),
            borderwidth=1
        )

        fig.show()
        return fig

    def export_monthly_table(self, language="en"):
        """
        Export a Plotly table of monthly + YTD returns to an interactive HTML file.
        """
        df = pd.DataFrame(self.monthly_returns)
        date_col = "month" if "month" in df.columns else "datetime"
        df["date"] = pd.to_datetime(df[date_col])
        df["year"] = df["date"].dt.year
        df["month_num"] = df["date"].dt.month

        month_labels = {"en":["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"],
                        "cn": ["1月","2月","3月","4月","5月","6月",
                        "7月","8月","9月","10月","11月","12月"]}
        other_labels = {
                    "en": {
                        'ytd_label': "YTD",
                        'year': "Year",
                    },
                    "cn": {
                        'ytd_label': "年初至今",
                        'year': "年分",
                    }
                }
        settings = {
                "en": {
                    'font': dict(family="Roboto", color="black", size=12),
                },
                "cn": {
                    'font': dict(family="Roboto", color="black", size=12),  
                }
            }

        if language == "en":
            table_labels = month_labels['en']
            ytd_label = other_labels['en']['ytd_label']
            year_label = other_labels['en']['year']
        elif language == "cn":
            table_labels = month_labels['cn']
            ytd_label = other_labels['cn']['ytd_label']
            year_label = other_labels['cn']['year']

        pivot = (
            df.pivot_table(index="year", columns="month_num", values="value", aggfunc="last")
              .reindex(columns=range(1, 13))
              .sort_index(ascending=False)
        )
        ytd = (pivot.add(1).prod(axis=1, skipna=True) - 1)
        pivot["YTD"] = ytd

        # format values
        def pct_str(x):
            return f"{x*100:,.2f}%" if pd.notna(x) else ""

        display_cols = table_labels + [ytd_label]
        pivot.columns = table_labels + [ytd_label]
        pivot = pivot[display_cols]
        pivot_str = pivot.applymap(pct_str)
        pivot_str.insert(0, year_label, pivot_str.index.astype(str))
        header_values = [year_label] + display_cols
        # ---------------
        num_rows = pivot_str.shape[0]
        num_cols = len(header_values)  # 14 (Year + 12 months + YTD)    
        
        px_per_char = 7.5
        month_label_width = max(52, int(px_per_char * max(len(lbl) for lbl in display_cols)))
        year_col_width   = max(64, int(px_per_char * len(year_label) + 10))

        # Optional: if values are long (e.g., many digits), widen month cells slightly
        # Look at a sample of formatted values to estimate content length
        try:
            sample_vals = pd.Series(
                [v for col in display_cols for v in pivot_str[col].head(5)]
            )
            max_val_chars = min(12, max((len(s) for s in sample_vals if isinstance(s, str)), default=6))
            month_label_width = max(month_label_width, int(px_per_char * max_val_chars) + 20)
        except Exception:
            pass

        # Compose column widths: first (year) + uniform month/YTD widths
        columnwidths = [year_col_width] + [month_label_width] * (num_cols - 1)

        # Base heights (px). With more rows, reduce height so tables don't get huge.
        base_cell_height = 28
        base_header_height = 30
        if num_rows > 10:
            # Scale down height roughly inversely with row count (cap at 14px)
            scale = 10 / num_rows
            base_cell_height = max(14, int(base_cell_height * scale))
            base_header_height = max(16, int(base_header_height * scale))

        # Enforce height:width ratio <= 10:26 (≈0.3846) using the NARROWEST column
        ratio_num, ratio_den = 10, 26
        min_col_width = min(columnwidths)
        max_allowed_height = int(min_col_width * ratio_num / ratio_den)

        cells_height = min(base_cell_height, max_allowed_height)
        header_height = min(base_header_height, max_allowed_height)

        # Safety: ensure at least 12px for legibility
        cells_height = max(12, cells_height)
        header_height = max(14, header_height)

        # ----------------------

        fig = go.Figure(
            data=[go.Table(
                columnwidth=[50] + [55]*13,
                header=dict(values=header_values,
                            fill_color="#cbb69d",
                            align="center",
                            font=settings[language]['font'],
                            height=30),
                cells=dict(values=[pivot_str[col].tolist() for col in pivot_str.columns],
                           fill_color="#f0f0f0",
                           font=settings[language]['font'],
                           align="center",
                           height=28)
            )]
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        fig.show()
        return fig
            
    def export_key_metrics_table(self, end_month, benchmark_fund=None, language="en", metrics=None, horizontal: bool = False, fix_aspect: bool = False):

        header_fill = "#cbb69d"
        cell_fill = "#f0f0f0"
        settings = {
            "en": {"font": dict(family="Roboto", color="black", size=12)},
            "cn": {"font": dict(family="Roboto", color="black", size=12)},
        }

        labels = {
            "en": {
                "metric": "Metric",
                "value": "Value",
                "cagr": "Annualized Return",
                "vol": "Volatility",
                "sharpe": "Sharpe Ratio",
                "sortino": "Sortino Ratio",
                "cum": "Cumulative Return",
                "mdd": "Max Drawdown",
                "beta": f'Beta to {benchmark_fund.name}',
                "corr": f"Correlation (vs. {benchmark_fund.name})",
                "win": "Win Rate (Monthly)",
                "best": "Best Month",
                "worst": "Worst Month",
                "aum": "AUM",
                "skew": "Skewness",
                "kurt": "Kurtosis",
                "turnover": "Avg. Monthly Turnover",
            },
            "cn": {
                "metric": "指标",
                "value": "数值",
                "cagr": "年复合增长率 (CAGR)",
                "vol": "波动率",
                "sharpe": "夏普比率",
                "sortino": "索提诺比率",
                "cum": "累計增长率",
                "mdd": "最大回撤",
                "beta": f"贝塔({benchmark_fund.name})",
                "corr": f"相关性（{benchmark_fund.name}）",
                "win": "月度胜率",
                "best": "最佳月份",
                "worst": "最差月份",
                "aum": "管理资产规模（AUM）",
                "skew": "偏度",
                "kurt": "峰度",
                "turnover": "平均月换手率",
            },
        }
        L = labels["en"] if language not in labels else labels[language]

        placeholder_values = {
            # Percentages
            "cagr": f"{100 * self.annualized_return(self.inception_date, end_month):.1f}%",
            "vol": f"{100 * self.volatility(self.inception_date, end_month):.1f}%",
            "mdd": f"{100 * self.max_drawdown(self.inception_date, end_month):.1f}%",
            "cum": f"{100 * self.cumulative_return(self.inception_date, end_month):.1f}%",
            "win": f"{100 * self.positive_months(self.inception_date, end_month):.1f}%",
            
            # Ratios
            "sharpe": f"{self.sharpe_ratio(self.inception_date, end_month):.2f}",
            "sortino": f"{self.sortino_ratio(self.inception_date, end_month):.2f}",
            "beta": f"{self.beta_to(benchmark_fund, self.inception_date, end_month):.2f}",
            "corr": f"{self.correlation_to(benchmark_fund, self.inception_date, end_month):.2f}",
            "win": f"{self.positive_months(self.inception_date, end_month)*100:.2f}%",
            "best": "[placeholder]", 
            "worst":"[placeholder]", 
            "aum": "[placeholder]", 
            "skew": "[placeholder]",
            "kurt": "[placeholder]", 
            "turnover": "[placeholder]",
        }

        default_order = ["cagr","vol","sharpe","sortino","mdd","beta","corr",
                        "win","best","worst","aum","skew","kurt","turnover"]

        if metrics is None:
            selected = default_order
        else:
            known = set(placeholder_values.keys())
            selected = [m for m in metrics if m in known]
            if not selected:
                raise ValueError("No valid metrics provided.")

        metric_labels = [L[k] for k in selected]
        metric_values = [placeholder_values[k] for k in selected]

        px_per_char = 7.5

        metric_width = max(120, int(px_per_char * max(len(str(s)) for s in metric_labels)) + 20)
        value_width  = max(100, int(px_per_char * max(len(str(s)) for s in metric_values)) + 20)
        n_rows = len(metric_labels)
        base_cell_h, base_header_h = 28, 32
        if n_rows > 12:
            scale = 12 / n_rows
            base_cell_h = max(16, int(base_cell_h * scale))
            base_header_h = max(18, int(base_header_h * scale))
        ratio_num, ratio_den = 10, 26
        min_col_width = min(metric_width, value_width)
        max_h = int(min_col_width * ratio_num / ratio_den)
        cell_h = max(14, min(base_cell_h, max_h))
        header_h = max(16, min(base_header_h, max_h))


        if not horizontal:
            fig = go.Figure(data=[go.Table(
                columnwidth=[metric_width, value_width],
                header=dict(values=[L["metric"], L["value"]],
                            fill_color=header_fill, align="left",
                            font=settings[language]["font"], height=header_h),
                cells=dict(values=[metric_labels, metric_values],
                        fill_color=cell_fill, align="left",
                        font=settings[language]["font"], height=cell_h),
            )])
            total_table_width = metric_width + value_width
            n_rows_rendered = n_rows + 1  # header + rows
        else:
            # --- FIXED: one list per column (each with a single value) ---
            col_widths = []
            for lab, val in zip(metric_labels, metric_values):
                w = max(len(lab), len(val))
                col_widths.append(max(90, int(px_per_char * w) + 16))
            total_table_width = sum(col_widths)
            n_rows_rendered = 2  # header + one value row

            fig = go.Figure(data=[go.Table(
                columnwidth=col_widths,
                header=dict(
                    values=metric_labels,
                    # bold=True,
                    fill_color=header_fill,
                    align=["center"] * len(metric_labels),
                    font=settings[language]["font"],
                    height=header_h,
                ),
                cells=dict(
                    # one column per metric, each column has a single row (the value)
                    values=[[v] for v in metric_values],
                    fill_color=cell_fill,
                    align=["center"] * len(metric_values),
                    font=settings[language]["font"],
                    height=cell_h,
                ),
            )])

        margins = dict(l=20, r=20, t=20, b=20)
        fig.update_layout(
            margin=margins,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        if fix_aspect:
        # Height:Width = aspect_ratio (H:W)
            ar_h, ar_w = (10, 50)
            # Derive canvas width from table width plus margins
            canvas_width = int(total_table_width + margins["l"] + margins["r"])
            canvas_height = int(canvas_width * (ar_h / ar_w))
            fig.update_layout(width=canvas_width, height=canvas_height)
        else:
            # Let Plotly auto-size width; set a sensible height from row count
            auto_height = int(margins["t"] + margins["b"] + header_h + cell_h * (n_rows_rendered - 1) + 8)
            fig.update_layout(height=auto_height)

        fig.show()
        return fig

    def summary_of_a_fund(self, benchmark_fund=None, language="en"):
        plot1 = self.export_monthly_table(language)
        plot2 = self.export_key_metrics_table(benchmark_fund=benchmark_fund, end_month=self.latest_date, language=language,metrics=["cagr","vol","sharpe","sortino","mdd","beta","corr","win"],horizontal=False)
        plot3 = self.plot_monthly_return_distribution()
        return plot1, plot2, plot3