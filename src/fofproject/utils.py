from typing import Union
from datetime import datetime
import pandas as pd

def parse_month(mstr: Union[str, datetime]) -> datetime:
    if isinstance(mstr, str):
        return datetime.strptime(mstr, "%Y-%m")
    return mstr

def list_of_dicts_to_df(lst, value_col_name):
    """Convert list of dicts to a DataFrame keyed by 'month'."""
    df = pd.DataFrame(lst)
    return df[["month", "value"]].rename(columns={"value": value_col_name})