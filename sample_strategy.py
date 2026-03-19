"""
Zip-ready ML strategy for the MIG Quant Competition.

Bundle this file together with `model_weights.npz`.

The model is a linear ridge regressor trained offline on features derived only
from the open-price matrix, so inference remains fully compliant with the
sandbox input contract.
"""

import os

import numpy as np


START_CASH = 25_000.0
FEE_RATE = 0.001
POSITION_LIMIT = 100

_MODEL_CACHE = None


def get_actions(prices: np.ndarray) -> np.ndarray:
    """
    Return an integer actions matrix with the same shape as `prices`.
    """
    global _MODEL_CACHE

    prices = np.asarray(prices, dtype=float)
    num_stocks, num_days = prices.shape
    actions = np.zeros((num_stocks, num_days), dtype=int)

    if _MODEL_CACHE is None:
        model_path = os.path.join(os.path.dirname(__file__), "model_weights.npz")
        data = np.load(model_path)
        _MODEL_CACHE = {
            "lookbacks": data["lookbacks"].astype(int),
            "ma_lookbacks": data["ma_lookbacks"].astype(int),
            "vol_lookbacks": data["vol_lookbacks"].astype(int),
            "coef": data["coef"].astype(float),
            "intercept": float(data["intercept"]),
            "mean": data["mean"].astype(float),
            "scale": np.where(data["scale"] == 0.0, 1.0, data["scale"]).astype(float),
            "top_k": int(data["top_k"]),
            "rebalance_every": int(data["rebalance_every"]),
            "gross": float(data["gross"]),
            "use_regime": bool(int(data["use_regime"])) if "use_regime" in data else False,
        }

    model = _MODEL_CACHE
    lookbacks = model["lookbacks"]
    ma_lookbacks = model["ma_lookbacks"]
    vol_lookbacks = model["vol_lookbacks"]
    coef = model["coef"]
    intercept = model["intercept"]
    mean = model["mean"]
    scale = model["scale"]
    top_k = model["top_k"]
    rebalance_every = model["rebalance_every"]
    gross = model["gross"]
    use_regime = model["use_regime"]

    start_day = int(np.max(lookbacks))
    regime_start_day = max(start_day, 100)
    if num_days <= regime_start_day:
        return actions

    cash = START_CASH
    positions = np.zeros(num_stocks, dtype=int)
    pending_buy_day = -1
    pending_target = np.zeros(num_stocks, dtype=int)
    pending_score = np.zeros(num_stocks, dtype=float)

    market_mean = prices.mean(axis=0)

    for day in range(regime_start_day, num_days):
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

        if (day - regime_start_day) % rebalance_every != 0:
            continue

        prev_prices = prices[:, day - 1]
        equity = cash + float(np.dot(positions, prev_prices))
        if equity <= 0:
            break

        feature_columns = []
        for lb in lookbacks:
            feature_columns.append(prev_prices / prices[:, day - lb] - 1.0)

        for lb in ma_lookbacks:
            moving_average = prices[:, day - lb : day].mean(axis=1)
            feature_columns.append(prev_prices / moving_average - 1.0)

        vol_cache = {}
        for lb in vol_lookbacks:
            volatility = np.std(np.diff(np.log(prices[:, day - lb : day]), axis=1), axis=1)
            vol_cache[int(lb)] = volatility
            feature_columns.append(volatility)

        rank_inputs = (
            feature_columns[list(lookbacks).index(20)],
            feature_columns[list(lookbacks).index(60)],
            -vol_cache[20],
        )
        for values in rank_inputs:
            order = np.argsort(values)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(num_stocks, dtype=float) / max(num_stocks - 1, 1)
            feature_columns.append(ranks)

        features = np.column_stack(feature_columns)
        features = (features - mean) / scale
        scores = features @ coef + intercept

        target_positions = np.zeros(num_stocks, dtype=int)
        day_prices = prices[:, day]
        regime_ok = True
        if use_regime:
            market_ma100 = market_mean[day - 100 : day].mean()
            market_ma50 = market_mean[day - 50 : day].mean()
            regime_ok = market_mean[day - 1] > market_ma100 and market_ma50 > market_ma100

        if regime_ok:
            leaders = np.argsort(scores)[-top_k:]
            vol20 = np.maximum(vol_cache[20], 1e-4)
            inv_vol = 1.0 / vol20[leaders]
            weights = inv_vol / inv_vol.sum()

            for weight, stock_idx in zip(weights, leaders):
                raw_shares = int((equity * gross * weight) / (day_prices[stock_idx] * (1.0 + FEE_RATE)))
                target_positions[stock_idx] = min(POSITION_LIMIT, max(0, raw_shares))

        delta = target_positions - positions
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
            pending_score = scores.copy()

    return actions
