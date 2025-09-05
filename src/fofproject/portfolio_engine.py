rebalance_paths(returns_df, weights_df, method: str="end") -> DataFrame[date, asset_id, begin_w, end_w, period_ret]

portfolio_returns(returns_df, weights_df) -> Series[date] — Σ(w_{t-1} * r_t)

contribution_by_asset(returns_df, weights_df) -> DataFrame[date, asset_id, contrib]

hit_ratio(returns: Series) -> float — % periods > 0

sortino(returns: Series, mar: float|Series=0.0, periods_per_year: int=12) -> float

skew_kurt(returns: Series) -> Tuple[float, float]

