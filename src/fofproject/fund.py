from datetime import datetime
from typing import List, Dict, Union
import plotly.graph_objects as go
import pandas as pd
import math
import numpy as np
from fofproject.utils import parse_month, list_of_dicts_to_df, hex_to_rgba

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
