from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import MockTrainingConfig
from .tick_env import TickTradingEnv

log = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    _HAS_SB3 = True
except Exception:  # pragma: no cover
    PPO = None
    DummyVecEnv = None
    _HAS_SB3 = False


class RandomAgent:
    def __init__(self, env: TickTradingEnv):
        self.action_space = 3
        self.env = env

    def learn(self, total_timesteps: int = 0):
        obs, _ = self.env.reset()
        reward_history = []
        for _ in range(total_timesteps or 1_000):
            action = np.random.randint(0, self.action_space)
            obs, reward, done, *_ = self.env.step(action)
            reward_history.append(reward)
            if done:
                obs, _ = self.env.reset()
        log.info("RandomAgent finished. mean reward=%.4f", np.mean(reward_history))
        return self

    def predict(self, obs, deterministic: bool = False):
        return np.array([np.random.randint(0, self.action_space)]), None


def train_rl_agent(envs: List[TickTradingEnv], cfg: MockTrainingConfig):
    """Train an RL agent (PPO when available, otherwise a RandomAgent baseline)."""
    if not envs:
        raise ValueError("At least one environment is required for RL training.")

    if _HAS_SB3:
        def make_env_fn(env: TickTradingEnv):
            def _init():
                return env

            return _init

        vec_env = DummyVecEnv([make_env_fn(env) for env in envs])
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=2048,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            ent_coef=cfg.ent_coef,
            gamma=cfg.gamma,
            seed=cfg.seed,
            verbose=1,
        )
        model.learn(total_timesteps=cfg.total_timesteps)
        return model
    else:
        log.warning("stable-baselines3 not found; falling back to RandomAgent.")
        agent = RandomAgent(envs[0])
        agent.learn(total_timesteps=cfg.total_timesteps)
        return agent
