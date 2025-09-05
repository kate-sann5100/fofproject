from typing import Dict
import pandas as pd
import plotly.graph_objects as go
from fofproject.fund import Fund
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
from fofproject.utils import hex_to_rgba

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
        "RDGFF": "#C1AE94",
        "MSCI CHINA": "#989A9C",
        "MSCI WORLD": "#DDDDDE",
    }
}



STYLE_DICT = {
    "default": {
        "palette": PALETTES["default"],
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
                    "y": -0.15 # scale to the entire plot area
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
        },
        "markers": {
            "enabled": True   # 👈 markers every ~10% of the data points   
        },
        "value_boxes": {
            "enabled": False,
            "format": "+.2%",
            "box": {
                "bgcolor": "rgba(255,255,255,0.9)",
                "borderwidth": 1,
                "font_size": 12,
                "xshift": 6,
                "yshift": 6
            },
            # Show only for lead fund every step and final, for others only final
            "rules": {
                "lead":   {"every_step": True,  "final": True},
                "others": {"every_step": False, "final": True}
            }
        }
                },
    "excel": {
        "palette": PALETTES["excel"],
        "layout_config": {
            "font": {
                "family": "Microsoft Yahei Light, Roboto, sans-serif",
                "size": 14,
                "color": "#2f2f2f"
            },
            "margin": {
                "l": 70,  # left margin
                "r": 20,  # right margin
                "t": 70,  # top margin
                "b": 110  # bottom margin   
            },
            "grid_color": "#DACEBF",
            "date_ticks": {
                "num": 6, # Number of date ticks to display
                "format": "%b %Y" # Format for date ticks
            },
            "annotation": {
                "add_annotation": False, # True if we want to add an annotation
                "position": {
                    "x": 0.5, # scale to the entire plot area
                    "y": -0.13 # scale to the entire plot area
                }
            },
            "legend": {
                "orientation": "h", # "h" for horizontal, "v" for vertical
                "position": {
                    "x": 0.5, # scale to the entire plot area
                    "y": -0.12 # scale to the entire plot area
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
        },
        "markers": {
            "enabled": False   # 👈 no markers for excel style
        },
        "value_boxes": {
            "enabled": True,
            "format": "+.2%",
            "box": {
                "bgcolor": "rgba(255,255,255,0.9)",
                "borderwidth": 1,
                "font_size": 12,
                "xshift": 6,
                "yshift": 6
            },
            # Show only for lead fund every step and final, for others only final
            "rules": {
                "lead":   {"every_step": True,  "final": True},
                "others": {"every_step": False, "final": True}
            }
        }
    },

}

def _add_value_boxes(fig, xs, ys, *, indices, color, fmt, boxcfg):
    for i in indices:
        fig.add_annotation(
            x=xs[i], y=ys[i],
            text=f"{ys[i]:{fmt}}",
            showarrow=False,
            bgcolor=hex_to_rgba(color, 0.15),   # transparent fill based on trace color
            bordercolor=color,
            borderwidth=boxcfg.get("borderwidth", 1),
            font=dict(size=boxcfg.get("font_size", 12), color=color),
            xanchor="left", yanchor="bottom", align="center",
            xshift=boxcfg.get("xshift", 6), yshift=boxcfg.get("yshift", 6),
            opacity=1.0,
        )


def plot_cumulative_returns(
        funds: Dict[str, Fund], 
        title: str, 
        start_month: str = None, # YYYY-MM
        end_month: str = None, # YYYY-MM
        style: str = "default",
        language: str = "en"  # "en" or "cn"
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
                hovertemplate="<b>%{fullData.name}</b><br>%{y:.2%}<extra></extra>",
                line=dict(
                    width=trace_config["line_width"]["lead"] if fund.name == lead_name else trace_config["line_width"]["other"], 
                    color=color, 
                    shape="spline", 
                    smoothing=0.6),
            )
        )

        # --- Add annotation ---
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1, y=-0.2,         
            text="<b>Strictly Confidential</b>" if language == "en" else "内部信息 严格密保",
            showarrow=False,
            font=dict(size=9, color="black"), 
            xanchor="right", yanchor="bottom", align="center",
            opacity=0.15,  
        )

        step = max(1, int(len(months) / 12))
        marker_indices = list(range(0, len(months), step))
        box_indices = list(range(0, len(months), step * 2))

        # Add markers for every ~10% of the data points
        if STYLE_DICT.get(style, STYLE_DICT["default"]).get("markers", {}).get("enabled", False):

            # always include the last point
            if (len(months) - 1) not in marker_indices:
                marker_indices.append(len(months) - 1)

            fig.add_trace(
                go.Scatter(
                    x=[months[i] for i in marker_indices],
                    y=[cumulative_returns[i] for i in marker_indices],
                    mode='markers',
                    showlegend=False,
                    hoverinfo='skip',
                    marker=dict(
                        size=trace_config["mark_size"]["lead"] if fund.name == lead_name else trace_config["mark_size"]["other"],
                        color=color,
                        line=dict(width=1, color="white")
                )
            )
        )
        # Add value boxes based on whether the trace is a lead or not / some style do not need value boxes
        value_box_cfg = STYLE_DICT.get(style, STYLE_DICT["default"]).get("value_boxes", {})
        if value_box_cfg.get("enabled", False):
            rules = value_box_cfg.get("rules", {
                "lead": {"every_step": True, "final": True},
                "others": {"every_step": False, "final": True}
            })
            is_lead = (fund.name == lead_name)

            idxs = []
            if is_lead and rules["lead"].get("every_step", True):
                # just copy box_indices for lead
                idxs.extend(box_indices)
            elif (not is_lead) and rules["others"].get("every_step", False):
                idxs.extend(box_indices)
            if (is_lead and rules["lead"].get("final", True)) or (not is_lead and rules["others"].get("final", True)):
                last = len(months) - 1
                if last not in idxs:
                    idxs.append(last)

            if idxs:  # draw
                _add_value_boxes(
                    fig,
                    months,
                    cumulative_returns,
                    indices=sorted(set(idxs)),
                    color=palettes.get(fund.name, DEFAULT_COLOR),
                    fmt=value_box_cfg.get("format", "+.2%"),
                    boxcfg=value_box_cfg.get("box", {})
                )
    # --------------------------- Set Layout ---------------------------
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>", 
            font=dict(size=32), 
            xanchor="left" if style == "excel" else "center", 
            x=0.15 if style == "excel" else 0.5, 
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

def plot_fund_correlation_heatmap(
    funds: dict,
    *,
    method: str = "pearson",      # "pearson" | "spearman" | "kendall"
    min_overlap: int = 6,         # require at least this many overlapping months for a correlation
    title: str = "Fund Return Correlations"
):
    """
    Build pairwise correlations of monthly returns using pairwise-complete data (max overlap)
    and plot as a heatmap. Returns (fig, corr_df, overlap_df).
    """
    # 1) Assemble a wide DataFrame of monthly returns aligned on month
    series = {}
    for name, f in funds.items():
        s = pd.Series(
            {e["month"]: float(e["value"]) for e in f.monthly_returns if e.get("value") is not None}
        ).sort_index()
        series[name] = s

    wide = pd.DataFrame(series)  # index: month; columns: fund names (may contain NaN for missing months)

    # 2) Pairwise overlap counts (n_ij) and correlations with pairwise-complete observations
    mask = wide.notna().astype(int)
    overlap = mask.T @ mask                        # n_ij = count of months present in both i and j
    corr = wide.corr(method=method, min_periods=min_overlap)

    # Ensure diagonals look tidy
    for c in corr.columns:
        corr.loc[c, c] = 1.0
        overlap.loc[c, c] = mask[c].sum()

    # 3) Build labels and hover text
    funds_order = list(corr.columns)
    z = corr.loc[funds_order, funds_order].values
    n_pairs = overlap.loc[funds_order, funds_order].values

    text = np.empty_like(z, dtype=object)
    hover = np.empty_like(z, dtype=object)
    for i, rname in enumerate(funds_order):
        for j, cname in enumerate(funds_order):
            r = z[i, j]
            n = int(n_pairs[i, j])
            if pd.isna(r):
                text[i, j] = "–"
                hover[i, j] = f"{rname} × {cname}<br>n = {n}<br>not enough overlap"
            else:
                text[i, j] = f"{r:.2f}\n(n={n})"
                hover[i, j] = f"{rname} × {cname}<br>ρ = {r:.3f}<br>n = {n}"

    # 4) Plotly heatmap (diverging scale, centered at 0)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=funds_order,
            y=funds_order,
            colorscale="RdBu",
            zmin=-1, zmax=1, zmid=0,
            colorbar=dict(title="ρ"),
            text=text,
            texttemplate="%{text}",
            hoverinfo="text",
            hovertext=hover
        )
    )

    # 5) Layout polish
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center"),
        template="plotly_white",
        font=dict(family="Montserrat, Roboto", size=13, color="#53565A"),
        margin=dict(l=80, r=40, t=80, b=80),
        xaxis=dict(showgrid=False, tickangle=45),
        yaxis=dict(showgrid=False, autorange="reversed")  # matrix-style orientation
    )

    fig.show()
    return fig, corr, overlap