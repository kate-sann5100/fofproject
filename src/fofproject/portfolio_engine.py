rebalance_paths(returns_df, weights_df, method: str="end") -> DataFrame[date, asset_id, begin_w, end_w, period_ret]

portfolio_returns(returns_df, weights_df) -> Series[date] — Σ(w_{t-1} * r_t)

contribution_by_asset(returns_df, weights_df) -> DataFrame[date, asset_id, contrib]

rollup_cumulative(returns: Series) -> Series — cumprod

max_drawdown(nav_or_cum: Series) -> Dict[peak_date, trough_date, mdd]

hit_ratio(returns: Series) -> float — % periods > 0

%_negative(returns: Series) -> float

value_at_risk(returns: Series, level: float=0.95, method: str="historical") -> float

sharpe(returns: Series, rf: Series|float=0.0, periods_per_year: int=12) -> float

sortino(returns: Series, mar: float|Series=0.0, periods_per_year: int=12) -> float

skew_kurt(returns: Series) -> Tuple[float, float]

correlation_matrix(returns_wide: DataFrame) -> DataFrame