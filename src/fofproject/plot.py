from typing import Dict
import pandas as pd
import plotly.graph_objects as go
from fofproject.fund import Fund

DEFAULT_COLOR = "#888888"

# palettes to decide colors for different funds based on themes
PALETTES = {
    "default": {
        "RDGFF": "#2F2F2F",
        "MSCI CHINA": "#D8C3A5",
        "MSCI WORLD": "#B8AEA0",
    },
    "dark": {
        "RDGFF": "#B58B80",
        "MSCI CHINA": "#DACEBF",
        "MSCI WORLD": "#C1AE94",
    },
    "excel": {
        "RDGFF": "#E0E0E0",
        "MSCI CHINA": "#A67C52",
        "MSCI WORLD": "#6A5ACD",
    }
}


LAYOUT_CONFIG = {
    "font": {
        "family": "Roboto, MontSerrat Semibold, sans-serif",
        "size": 14,
        "color": "#2f2f2f"
    },
    "margin": {
        "l": 70,  # left margin
        "r": 20,  # right margin
        "t": 70,  # top margin
        "b": 110  # bottom margin   
    },
    "grid_color": "#E9E9E9",
    "date_ticks": {
        "num": 6, # Number of date ticks to display
        "format": "%b %Y" # Format for date ticks
    },
    "annotation": {
        "add_annotation": True, # True if we want to add an annotation
        "position": {
            "x": 0.5, # scale to the entire plot area
            "y": -0.13 # scale to the entire plot area
        }
    },
    "legend": {
        "orientation": "v", # "h" for horizontal, "v" for vertical
        "position": {
            "x": 0.0, # scale to the entire plot area
            "y": -0.1 # scale to the entire plot area
        }
    },
}

TRACE_CONFIG = {
    "mark_size": {
        "lead": 12, # marker size for the lead fund
        "other": 8 # marker size for the other funds
    },
    "line_width": {
        "lead": 3, # line width for the lead fund
        "other": 2 # line width for the other funds
    }
}

def plot_cumulative_returns(
        funds: Dict[str, Fund], 
        title: str, 
        start_date: str = None,
        end_date: str = None,
        palettes="default",
        layout_config=LAYOUT_CONFIG,
        trace_config=TRACE_CONFIG
    ):
    
    palettes = PALETTES[palettes]
    
    # Parsing the date strings into date objects

    from datetime import datetime, date
    def _to_date(v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return datetime.strptime(v, fmt).date()
            except (TypeError, ValueError):
                pass
        raise ValueError(f"Invalid date '{v}'. Expected dd/mm/yy")
        # Collect all dates across funds
    all_dates = []
    per_fund_first_dates = []

    for f in funds.values():
        dates = [_to_date(e["date"]) for e in f.monthly_returns]
        if not dates:
            raise ValueError("No dates found in funds.monthly_returns")
        dates.sort()
        all_dates.extend(dates)
        per_fund_first_dates.append(dates[0])   # first available date for this fund

    # Global bounds across ALL data points (unchanged)
    data_min, data_max = min(all_dates), max(all_dates)

    # Latest start across funds (the overlap-friendly start)
    latest_common_start = max(per_fund_first_dates)

    # Respect a user-specified start_date but don't allow starting before the latest_common_start
    # (so we guarantee every fund has data from 'start' onward)
    user_start = _to_date(start_date)
    start = max(user_start or data_min, latest_common_start)

    # End date behavior as you had it (you only asked to unify the start)
    end = _to_date(end_date) or data_max

    if start > end:
        raise ValueError("start_date must be <= end_date")


    # Identify lead fund
    lead_name = "RDGFF" if "RDGFF" in list(funds.keys()) else (list(funds.keys())[0])

    # Initialise figure
    fig = go.Figure()

    # --------------------------- Plot Trace ---------------------------

    # Dict storing final cumulative returns for each fund
    final_cumulative_returns = {}
    for fund in funds.values():
        # Get dates within the selected range (inclusive), normalizing to date objects
        dates = []
        for e in fund.monthly_returns:
            d = e['date']
            if isinstance(d, datetime):
                d_norm = d.date()
            elif isinstance(d, date):
                d_norm = d
            else:
                d_norm = _to_date(d)
            if start <= d_norm <= end:
                dates.append(d_norm)
        # Compute cumulative returns for each month
        cumulative_returns = [fund.cumulative_return(start_month=dates[0].strftime('%Y-%m'), end_month=date.strftime('%Y-%m')) -1 for date in dates]
        # Update final_cumulative_returns
        final_cumulative_returns[fund.name] = cumulative_returns[-1]
        # Get color for the fund
        color = palettes.get(fund.name, DEFAULT_COLOR)
        # Add trace for the fund
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode="lines",
                name=fund.name,
                hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
                line=dict(
                    width=trace_config["line_width"]["lead"] if fund.name == lead_name else trace_config["line_width"]["other"], 
                    color=color, 
                    shape="spline", 
                    smoothing=0.6),
            )
        )
        # Add markers for every ~10% of the data points
        step = max(1, int(len(dates) / 10))
        fig.add_trace(
            go.Scatter(
                x=dates[::step], y=cumulative_returns[::step], mode='markers',showlegend=False,hoverinfo='skip',
            marker=dict(size=trace_config["mark_size"]["lead"] if fund.name == lead_name else trace_config["mark_size"]["other"],
                color=color, 
                line=dict(width=1, color="white"))
            )
        )
    # --------------------------- Set Layout ---------------------------
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>", 
            font=dict(size=32), 
            xanchor="center", 
            x=0.5, 
            yanchor="middle", 
            y=0.95
        ),
        template="plotly_white",
        font=layout_config["font"],
        margin=layout_config["margin"],
        xaxis=dict(
            showgrid=True,
            gridcolor=layout_config["grid_color"],
            tickformat=layout_config["date_ticks"]["format"],
            tickvals=[dt for dt in list(dates)[::max(1, len(dates)//6)]],
            ticktext=[
                dt.strftime(layout_config["date_ticks"]["format"]) 
                for dt in list(dates)[::max(1, len(dates)//layout_config["date_ticks"]["num"])]
            ],
        ),
        yaxis=dict(title="Cumulative Return (%)", tickformat=".0%", zeroline=True),
        legend=dict(
            orientation=layout_config["legend"]["orientation"],
            yanchor="middle",
            y=layout_config["legend"]["position"]["y"],
            xanchor="center",
            x=layout_config["legend"]["position"]["x"]
        ),
        hovermode="x unified",
    )

    # --------------------------- Add annotation ---------------------------
    if layout_config["annotation"]["add_annotation"]:
        # Initialise summary lines, a list storing all annotation strings
        summary_lines = []
        # Add lead fund annotation as the first entry
        if lead_name in final_cumulative_returns:
            fcr = final_cumulative_returns.pop(lead_name) # pop returns the value of the key and delete the key from the dict
            summary_lines.append(f"<span style='color:{palettes.get(lead_name, DEFAULT_COLOR)};'><b>{lead_name}: {fcr:+.2%}</b></span>")
        # Add other funds annotations
        summary_lines.append(
            " &nbsp; &nbsp; ".join(
                [
                    f"<b><span style='color:{palettes.get(k, DEFAULT_COLOR)};'>{k}</span>: {v:+.2%}</b>" 
                    for k, v in final_cumulative_returns.items()
                ]
            )
        )
        # Add annotations to the figure
        if summary_lines:
            fig.add_annotation(
                xref="paper", yref="paper",
                x=layout_config["annotation"]["position"]["x"], 
                y=layout_config["annotation"]["position"]["y"],
                showarrow=False,
                text="<br>".join(summary_lines)
            )
    # --------------------------- Show figure ---------------------------
    fig.show()
    
    return fig
