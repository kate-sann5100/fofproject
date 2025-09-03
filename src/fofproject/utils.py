from typing import Union
from datetime import datetime

def parse_month(mstr: Union[str, datetime]) -> datetime:
    if isinstance(mstr, str):
        return datetime.strptime(mstr, "%Y-%m")
    return mstr