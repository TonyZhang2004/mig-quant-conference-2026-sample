"""
Submission-ready strategy for the MIG Quant Competition.

This implementation is intentionally self-contained:
- Uses only `numpy`
- Avoids lookahead bias by trading on day `t` from information through `t - 1`
- Respects the documented 100-share position cap
- Sizes longs with an internal cash-aware simulator so actions stay feasible

Strategy summary
----------------
Long-only cross-sectional momentum with low turnover:
- Rank stocks by a blend of 90-day and 180-day momentum
- Skip the most recent 5 trading days to reduce short-term reversal noise
- Add a small 20-day trend / breakout component
- Volatility-adjust the final score and size positions inverse-vol
- Rebalance every 20 trading days into the top 4 names
- Stage rebalances as "sell today, buy tomorrow" so the action matrix stays
  compatible with a row-ordered backtester

The goal is robustness, not cleverness. This keeps the logic stable across
datasets while reducing fee drag from over-trading.
"""

import numpy as np


FEE_RATE = 0.001
START_CASH = 25_000.0
POSITION_LIMIT = 100

PRIMARY_LOOKBACK = 90
SECONDARY_LOOKBACK = 180
SHORT_LOOKBACK = 20
SKIP_RECENT_DAYS = 5
TOP_K = 4
REBALANCE_EVERY = 20
TARGET_GROSS_EXPOSURE = 0.95


def get_actions(prices: np.ndarray) -> np.ndarray:
    """
    Build an actions matrix from an anonymized open-price matrix.

    Parameters
    ----------
    prices : np.ndarray
        Shape (num_stocks, num_days). Each row is one stock and each column is a
        trading day in chronological order.

    Returns
    -------
    np.ndarray
        Integer trade matrix with the same shape as `prices`.
    """
    prices = np.asarray(prices, dtype=float)
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=int)

    # No signal is available until all lookback windows are populated.
    start_day = max(
        PRIMARY_LOOKBACK + SKIP_RECENT_DAYS,
        SECONDARY_LOOKBACK + SKIP_RECENT_DAYS,
        SHORT_LOOKBACK,
    )
    if num_days <= start_day:
        return actions

    cash = START_CASH
    positions = np.zeros(num_stocks, dtype=int)
    pending_buy_day = -1
    pending_target = np.zeros(num_stocks, dtype=int)
    pending_score = np.zeros(num_stocks, dtype=float)

    for day in range(start_day, num_days):
        if pending_buy_day == day:
            buy_indices = np.flatnonzero(pending_target > positions)
            if buy_indices.size:
                buy_order = buy_indices[np.argsort(-pending_score[buy_indices])]
                day_prices = prices[:, day]
                for stock_idx in buy_order:
                    buy_qty = pending_target[stock_idx] - positions[stock_idx]
                    buy_qty = min(buy_qty, POSITION_LIMIT - positions[stock_idx])
                    affordable = int(cash / (day_prices[stock_idx] * (1.0 + FEE_RATE)))
                    buy_qty = min(buy_qty, affordable)
                    if buy_qty <= 0:
                        continue
                    actions[stock_idx, day] += buy_qty
                    cash -= day_prices[stock_idx] * buy_qty * (1.0 + FEE_RATE)
                    positions[stock_idx] += buy_qty
            pending_buy_day = -1

        if (day - start_day) % REBALANCE_EVERY != 0:
            continue

        prev_prices = prices[:, day - 1]
        equity = cash + float(np.dot(positions, prev_prices))
        if equity <= 0:
            break

        primary_momentum = (
            prices[:, day - SKIP_RECENT_DAYS - 1]
            / prices[:, day - PRIMARY_LOOKBACK - SKIP_RECENT_DAYS]
            - 1.0
        )
        secondary_momentum = (
            prices[:, day - SKIP_RECENT_DAYS - 1]
            / prices[:, day - SECONDARY_LOOKBACK - SKIP_RECENT_DAYS]
            - 1.0
        )
        short_momentum = prices[:, day - 1] / prices[:, day - SHORT_LOOKBACK] - 1.0

        short_window = prices[:, day - SHORT_LOOKBACK : day]
        short_ma = short_window.mean(axis=1)
        breakout = prev_prices / short_ma - 1.0

        log_returns = np.diff(np.log(short_window), axis=1)
        vol = np.std(log_returns, axis=1)
        vol = np.maximum(vol, 1e-4)

        primary_order = np.argsort(primary_momentum)
        primary_ranks = np.empty_like(primary_order, dtype=float)
        primary_ranks[primary_order] = np.arange(num_stocks, dtype=float)

        secondary_order = np.argsort(secondary_momentum)
        secondary_ranks = np.empty_like(secondary_order, dtype=float)
        secondary_ranks[secondary_order] = np.arange(num_stocks, dtype=float)

        short_order = np.argsort(short_momentum)
        short_ranks = np.empty_like(short_order, dtype=float)
        short_ranks[short_order] = np.arange(num_stocks, dtype=float)

        breakout_order = np.argsort(breakout)
        breakout_ranks = np.empty_like(breakout_order, dtype=float)
        breakout_ranks[breakout_order] = np.arange(num_stocks, dtype=float)

        score = (
            0.45 * primary_ranks
            + 0.30 * secondary_ranks
            + 0.10 * short_ranks
            + 0.15 * breakout_ranks
        ) / vol

        leaders = np.argsort(score)[-TOP_K:]
        leader_vol = vol[leaders]
        inv_vol = 1.0 / leader_vol
        weights = inv_vol / inv_vol.sum()

        target_positions = np.zeros(num_stocks, dtype=int)
        day_prices = prices[:, day]
        gross_budget = equity * TARGET_GROSS_EXPOSURE

        for weight, stock_idx in zip(weights, leaders):
            share_budget = gross_budget * weight
            raw_shares = int(share_budget / (day_prices[stock_idx] * (1.0 + FEE_RATE)))
            target_positions[stock_idx] = min(POSITION_LIMIT, max(0, raw_shares))

        delta = target_positions - positions

        # Free cash before adding new risk.
        sell_indices = np.flatnonzero(delta < 0)
        for stock_idx in sell_indices:
            sell_qty = min(-delta[stock_idx], positions[stock_idx])
            if sell_qty <= 0:
                continue
            actions[stock_idx, day] -= sell_qty
            cash += day_prices[stock_idx] * sell_qty * (1.0 - FEE_RATE)
            positions[stock_idx] -= sell_qty

        if day + 1 < num_days:
            pending_buy_day = day + 1
            pending_target = target_positions.copy()
            pending_score = score.copy()

    return actions
