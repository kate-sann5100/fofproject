from typing import Dict
import pandas as pd
import plotly.graph_objects as go
from fofproject.fund import Fund
import datetime as dt
from dateutil.relativedelta import relativedelta

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

STYLE_DICT = {
    "default": {
        "palette": PALETTES["excel"],
        "layout_config": {
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
                    "y": -0.1 # scale to the entire plot area
                }
            },
            "legend": {
                "orientation": "v", # "h" for horizontal, "v" for vertical
                "position": {
                    "x": 0.0, # scale to the entire plot area
                    "y": -0.1 # scale to the entire plot area
                }
            },
        },
        "trace_config": {
            "mark_size": {
                "lead": 12, # marker size for the lead fund
                "other": 8 # marker size for the other funds
            },
            "line_width": {
                "lead": 3, # line width for the lead fund
                "other": 2 # line width for the other funds
            }
        }
    },
    "excel": {
        "palette": PALETTES["excel"],
        "layout_config": {
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
                    "y": -0.1 # scale to the entire plot area
                }
            },
            "legend": {
                "orientation": "v", # "h" for horizontal, "v" for vertical
                "position": {
                    "x": 0.0, # scale to the entire plot area
                    "y": -0.1 # scale to the entire plot area
                }
            },
        },
        "trace_config": {
            "mark_size": {
                "lead": 12, # marker size for the lead fund
                "other": 8 # marker size for the other funds
            },
            "line_width": {
                "lead": 3, # line width for the lead fund
                "other": 2 # line width for the other funds
            }
        }
    },
}

def plot_cumulative_returns(
        funds: Dict[str, Fund], 
        title: str, 
        start_month: str = None, # YYYY-MM
        end_month: str = None, # YYYY-MM
        style: str = "default"
    ):

    palettes = STYLE_DICT.get(style, STYLE_DICT["default"])["palette"]
    layout_config = STYLE_DICT.get(style, STYLE_DICT["default"])["layout_config"]
    trace_config = STYLE_DICT.get(style, STYLE_DICT["default"])["trace_config"]

    # Convert start and end month to datetime
    start_month = dt.datetime.strptime(start_month, '%Y-%m') if start_month else None
    end_month = dt.datetime.strptime(end_month, '%Y-%m') if end_month else None

    # Collect all months across funds
    start_months = [f.monthly_returns[0]["month"] for f in funds.values()]
    end_months = [f.monthly_returns[-1]["month"] for f in funds.values()]

    # Update start_month and end_month if the current months are not valid
    if not (start_month and start_month > min(start_months) and start_month < max(end_months)):
        start_month = max(start_months)
    if not (end_month and min(start_months) < end_month < max(end_months)):
        end_month = min(end_months)
    # Assert start month is before end month
    if start_month > end_month:
        raise ValueError("start_month must be <= end_month")
    # Get one month before start month
    prev_month = start_month - relativedelta(months=1) # get previous month

    # Identify lead fund
    lead_name = "RDGFF" if "RDGFF" in list(funds.keys()) else (list(funds.keys())[0])

    # Initialise figure
    fig = go.Figure()

    # --------------------------- Plot Trace ---------------------------

    # Dict storing final cumulative returns for each fund
    final_cumulative_returns = {}
    for fund in funds.values():
        # Compute cumulative returns for each month
        months = [
            entry["month"] 
            for entry in fund.monthly_returns 
            if start_month <= entry["month"] <= end_month
        ]
        cumulative_returns = [
            fund.cumulative_return(
                start_month=start_month, 
                end_month=entry["month"]
            ) 
            for entry in fund.monthly_returns
            if start_month <= entry["month"] <= end_month
        ]
        # Add one month before
        months = [prev_month] + months
        cumulative_returns = [0.0] + cumulative_returns
        # Update final_cumulative_returns
        final_cumulative_returns[fund.name] = cumulative_returns[-1]
        # Get color for the fund
        color = palettes.get(fund.name, DEFAULT_COLOR)
        # Add trace for the fund
        fig.add_trace(
            go.Scatter(
                x=months,
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
        step = max(1, int(len(months) / 10))
        fig.add_trace(
            go.Scatter(
                x=months[::step] + [months[-1]], 
                y=cumulative_returns[::step] + [cumulative_returns[-1]], 
                mode='markers',
                showlegend=False,
                hoverinfo='skip',
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
            tickvals=[dt for dt in list(months)[::max(1, len(months)//6)]],
            ticktext=[
                dt.strftime(layout_config["date_ticks"]["format"]) 
                for dt in list(months)[::max(1, len(months)//layout_config["date_ticks"]["num"])]
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
