from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import gymnasium as gym

    BaseEnv = gym.Env
except Exception:  # fallback lightweight base class
    gym = None

    class BaseEnv:  # type: ignore
        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError


class TickTradingEnv(BaseEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        window: int,
        fees: Dict[str, float],
        slippage_bps: float,
        max_position: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.data = data
        self.prices = prices
        self.window = window
        self.pos = 0.0
        self.cash = 1.0
        self.index = window
        self.max_index = data.shape[0] - 1
        self.fees = fees
        self.slippage = slippage_bps / 1e4
        self.max_position = max_position
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.index = self.window
        self.pos = 0.0
        self.cash = 1.0
        obs = self.data[self.index - self.window : self.index]
        return obs, {}

    def step(self, action: int):
        price = float(self.prices[self.index])
        fill_price = price * (1 + self.slippage * np.sign(action - 1))

        reward = 0.0
        done = False
        trade_flag = False

        if action == 1:  # buy
            trade_flag = True
            if self.pos < self.max_position:
                self.pos += 0.5
                self.cash -= fill_price * (1 + self.fees["buy"])
        elif action == 2:  # sell
            trade_flag = True
            if self.pos > -self.max_position:
                self.pos -= 0.5
                proceeds = fill_price * (1 - self.fees["sell"] - self.fees.get("tax_sell", 0))
                self.cash += proceeds

        portfolio = self.cash + self.pos * price
        reward = portfolio - 1.0

        self.index += 1
        if self.index >= self.max_index:
            done = True
        obs = self.data[self.index - self.window : self.index]
        info = {"portfolio": portfolio, "trade": trade_flag}
        return obs, reward, done, False, info


def make_vec_envs(
    arrays: List[np.ndarray],
    prices: List[np.ndarray],
    window: int,
    fees: Dict[str, float],
    slippage_bps: float,
) -> List[TickTradingEnv]:
    envs = [
        TickTradingEnv(array, price, window=window, fees=fees, slippage_bps=slippage_bps, seed=i)
        for i, (array, price) in enumerate(zip(arrays, prices))
    ]
    return envs
