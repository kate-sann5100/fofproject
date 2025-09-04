from typing import Union
from datetime import datetime

def parse_month(mstr: Union[str, datetime]) -> datetime:
    if isinstance(mstr, str):
        return datetime.strptime(mstr, "%Y-%m")
    return mstr

def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex color like '#RRGGBB' to rgba(R,G,B,alpha)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
