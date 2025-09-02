estimate_moments(returns_wide: DataFrame, method: str="hist") -> Tuple[mu:Series, cov:DataFrame]

mean_variance_opt(mu, cov, target: str="sharpe", bounds: Dict|tuple=(0,1), budget: float=1.0, constraints: Dict=None) -> Series[weight]

risk_parity(cov: DataFrame, bounds=(0,1)) -> Series[weight]

rebalance_to_optimal(current_weights: Series, target_weights: Series, turnover_limit: float=None) -> DataFrame[asset_id, trade_w]