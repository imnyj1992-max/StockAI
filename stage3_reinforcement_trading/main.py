"""Stage 3: Reinforcement learning driven virtual trading simulator."""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


Action = int
State = Tuple[int, ...]


@dataclass
class StepResult:
    """Container for environment step data to simplify GUI updates."""

    state: State
    reward: float
    done: bool
    step: int
    price: float
    cash: float
    equity: float
    position: int
    epsilon: float
    action_name: str
    info: str


class TradingEnvironment:
    """Random-walk price environment with a single long position constraint."""

    ACTIONS: List[str] = ["HOLD", "BUY", "SELL"]

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        episode_length: int = 200,
        window_size: int = 3,
        drift: float = 0.0005,
        volatility: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("초기 현금은 0보다 커야 합니다.")
        if episode_length <= 0:
            raise ValueError("에피소드 스텝 수는 1 이상이어야 합니다.")
        if window_size <= 0:
            raise ValueError("상태 윈도우 크기는 1 이상이어야 합니다.")

        self.initial_cash = float(initial_cash)
        self.episode_length = int(episode_length)
        self.window_size = int(window_size)
        self.drift = float(drift)
        self.volatility = float(volatility)
        self.rng = np.random.default_rng(seed)

        self.price: float = 100.0
        self.step_count: int = 0
        self.position: int = 0
        self.entry_price: float = 0.0
        self.cash: float = self.initial_cash
        self.total_reward: float = 0.0
        self._history: Deque[int] = deque([0] * self.window_size, maxlen=self.window_size)

    def reset(self) -> State:
        """Reset the environment for a new episode."""

        self.price = float(self.rng.uniform(80, 120))
        self.step_count = 0
        self.position = 0
        self.entry_price = 0.0
        self.cash = self.initial_cash
        self.total_reward = 0.0
        self._history = deque([0] * self.window_size, maxlen=self.window_size)
        return self._get_state()

    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, str]]:
        """Advance the environment by one tick."""

        if action < 0 or action >= len(self.ACTIONS):
            raise ValueError("지원하지 않는 액션입니다.")

        prev_price = self.price
        prev_equity = self._equity(prev_price)

        shock = float(self.rng.normal(self.drift, self.volatility))
        next_price = max(0.01, prev_price * (1.0 + shock))
        self.price = next_price

        reward = 0.0
        info: Dict[str, str] = {}

        # Execute action
        if action == 1:  # BUY
            if self.position == 0:
                if self.cash >= self.price:
                    self.position = 1
                    self.entry_price = self.price
                    self.cash -= self.price
                    info["trade"] = f"BUY @ {self.price:.2f}"
                else:
                    reward -= 1.0
                    info["trade"] = "BUY_FAILED_CASH"
            else:
                reward -= 0.5
                info["trade"] = "BUY_BLOCKED"
        elif action == 2:  # SELL
            if self.position == 1:
                self.position = 0
                profit = self.price - self.entry_price
                self.cash += self.price
                reward += profit
                info["trade"] = f"SELL @ {self.price:.2f} (P&L {profit:.2f})"
            else:
                reward -= 1.0
                info["trade"] = "SELL_BLOCKED"
        else:
            info["trade"] = "HOLD"

        # Equity-based reward component (unrealised P&L)
        new_equity = self._equity(self.price)
        reward += new_equity - prev_equity
        self.total_reward += reward

        pct_change = (self.price - prev_price) / prev_price
        self._history.append(self._discretise_return(pct_change))

        self.step_count += 1
        done = self.step_count >= self.episode_length

        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _discretise_return(self, value: float) -> int:
        threshold = 0.001
        if value > threshold:
            return 1
        if value < -threshold:
            return -1
        return 0

    def _get_state(self) -> State:
        return tuple(list(self._history) + [self.position])

    def _equity(self, price: float) -> float:
        return self.cash + self.position * price

    def metrics(self) -> Dict[str, float]:
        return {
            "price": self.price,
            "cash": self.cash,
            "equity": self._equity(self.price),
            "position": float(self.position),
            "total_reward": self.total_reward,
            "step": float(self.step_count),
        }


class QLearningAgent:
    """Tabular Q-learning agent with ε-greedy policy."""

    def __init__(
        self,
        action_count: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.98,
        min_epsilon: float = 0.05,
    ) -> None:
        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(self.action_count, dtype=np.float32))

    def select_action(self, state: State) -> Action:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_count))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> None:
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def on_episode_end(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class ReinforcementWindow(QMainWindow):
    """PyQt5 window orchestrating environment steps and agent updates."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StockAI - Stage 3: Reinforcement Learning Trader")
        self.resize(820, 600)

        self.env: TradingEnvironment | None = None
        self.agent: QLearningAgent | None = None
        self.current_state: State | None = None
        self.episode: int = 0

        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self._on_timer)

        self._build_ui()
        self._update_status_labels(reset=True)

    # ------------------------------------------------------------------
    # UI building
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        config_layout = QGridLayout()
        main_layout.addLayout(config_layout)

        self.cash_input = QLineEdit("10000")
        self.episode_input = QLineEdit("200")
        self.window_input = QLineEdit("3")
        self.alpha_input = QLineEdit("0.1")
        self.gamma_input = QLineEdit("0.95")
        self.epsilon_input = QLineEdit("0.2")
        self.decay_input = QLineEdit("0.98")
        self.min_epsilon_input = QLineEdit("0.05")

        labels = [
            ("초기 현금", self.cash_input),
            ("에피소드 스텝", self.episode_input),
            ("상태 윈도우", self.window_input),
            ("학습률 α", self.alpha_input),
            ("할인율 γ", self.gamma_input),
            ("탐험률 ε", self.epsilon_input),
            ("ε 감소율", self.decay_input),
            ("최소 ε", self.min_epsilon_input),
        ]
        for row, (text, widget) in enumerate(labels):
            config_layout.addWidget(QLabel(text), row, 0)
            config_layout.addWidget(widget, row, 1)

        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        self.start_button = QPushButton("시작")
        self.pause_button = QPushButton("일시정지")
        self.reset_button = QPushButton("리셋")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reset_button)

        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(self.pause_training)
        self.reset_button.clicked.connect(self.reset_training)

        status_layout = QGridLayout()
        main_layout.addLayout(status_layout)

        self.episode_label = QLabel("0")
        self.step_label = QLabel("0")
        self.price_label = QLabel("0.00")
        self.position_label = QLabel("0")
        self.cash_label = QLabel("0.00")
        self.equity_label = QLabel("0.00")
        self.reward_label = QLabel("0.00")
        self.epsilon_label = QLabel("0.00")
        self.action_label = QLabel("HOLD")
        self.trade_label = QLabel("-")

        status_items = [
            ("에피소드", self.episode_label),
            ("스텝", self.step_label),
            ("가격", self.price_label),
            ("포지션", self.position_label),
            ("현금", self.cash_label),
            ("자산가치", self.equity_label),
            ("직전 보상", self.reward_label),
            ("ε", self.epsilon_label),
            ("액션", self.action_label),
            ("거래", self.trade_label),
        ]
        for index, (text, widget) in enumerate(status_items):
            row = index // 2
            column = (index % 2) * 2
            status_layout.addWidget(QLabel(text), row, column)
            status_layout.addWidget(widget, row, column + 1)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.NoWrap)
        main_layout.addWidget(self.log_view)

    # ------------------------------------------------------------------
    # Training control logic
    # ------------------------------------------------------------------
    def start_training(self) -> None:
        if self.env is None or self.agent is None or self.current_state is None:
            try:
                self._initialise_environment()
            except ValueError as exc:
                QMessageBox.critical(self, "설정 오류", str(exc))
                return

        if not self.timer.isActive():
            self.timer.start()
            self.log_view.append("학습을 시작합니다.")

    def pause_training(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.log_view.append("학습을 일시정지했습니다.")

    def reset_training(self) -> None:
        self.timer.stop()
        try:
            self._initialise_environment()
        except ValueError as exc:
            QMessageBox.critical(self, "설정 오류", str(exc))
            return
        self.log_view.clear()
        self.log_view.append("환경과 에이전트를 리셋했습니다.")

    def _initialise_environment(self) -> None:
        initial_cash = self._parse_float(self.cash_input.text(), "초기 현금")
        episode_steps = self._parse_int(self.episode_input.text(), "에피소드 스텝")
        window_size = self._parse_int(self.window_input.text(), "상태 윈도우")
        alpha = self._parse_float(self.alpha_input.text(), "학습률 α", positive=True)
        gamma = self._parse_float(self.gamma_input.text(), "할인율 γ", positive=True)
        epsilon = self._parse_float(self.epsilon_input.text(), "탐험률 ε", positive=True)
        decay = self._parse_float(self.decay_input.text(), "ε 감소율", positive=True)
        min_epsilon = self._parse_float(self.min_epsilon_input.text(), "최소 ε", positive=True)

        if not (0.0 < gamma <= 1.0):
            raise ValueError("할인율 γ는 0과 1 사이여야 합니다.")
        if not (0.0 < epsilon <= 1.0):
            raise ValueError("탐험률 ε는 0과 1 사이여야 합니다.")
        if not (0.0 < decay <= 1.0):
            raise ValueError("ε 감소율은 0과 1 사이여야 합니다.")
        if not (0.0 <= min_epsilon <= 1.0):
            raise ValueError("최소 ε는 0 이상 1 이하이어야 합니다.")

        self.env = TradingEnvironment(
            initial_cash=initial_cash,
            episode_length=episode_steps,
            window_size=window_size,
        )
        self.agent = QLearningAgent(
            action_count=len(TradingEnvironment.ACTIONS),
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=decay,
            min_epsilon=min_epsilon,
        )
        self.current_state = self.env.reset()
        self.episode = 1
        self._update_status_labels(reset=True)

    def _on_timer(self) -> None:
        if self.env is None or self.agent is None or self.current_state is None:
            self.timer.stop()
            return

        action = self.agent.select_action(self.current_state)
        next_state, reward, done, info = self.env.step(action)
        self.agent.update(self.current_state, action, reward, next_state, done)

        metrics = self.env.metrics()
        self._update_status_labels(
            StepResult(
                state=next_state,
                reward=reward,
                done=done,
                step=int(metrics["step"]),
                price=metrics["price"],
                cash=metrics["cash"],
                equity=metrics["equity"],
                position=int(metrics["position"]),
                epsilon=self.agent.epsilon,
                action_name=TradingEnvironment.ACTIONS[action],
                info=info.get("trade", ""),
            )
        )

        log_line = (
            f"Episode {self.episode} | Step {int(metrics['step']):03d} | "
            f"Price {metrics['price']:.2f} | Action {TradingEnvironment.ACTIONS[action]} | "
            f"Reward {reward:.2f} | Equity {metrics['equity']:.2f}"
        )
        if info.get("trade"):
            log_line += f" | {info['trade']}"
        self.log_view.append(log_line)
        self.log_view.ensureCursorVisible()

        self.current_state = next_state

        if done:
            episode_reward = self.env.total_reward
            self.log_view.append(
                f"에피소드 {self.episode} 종료 - 총 보상 {episode_reward:.2f}, ε={self.agent.epsilon:.3f}"
            )
            self.agent.on_episode_end()
            self.episode += 1
            self.current_state = self.env.reset()
            self._update_status_labels(reset=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _update_status_labels(self, result: StepResult | None = None, reset: bool = False) -> None:
        if reset or result is None:
            self.episode_label.setText(str(self.episode))
            self.step_label.setText("0")
            if self.env is not None:
                metrics = self.env.metrics()
                self.price_label.setText(f"{metrics['price']:.2f}")
                self.position_label.setText(str(int(metrics["position"])))
                self.cash_label.setText(f"{metrics['cash']:.2f}")
                self.equity_label.setText(f"{metrics['equity']:.2f}")
                self.reward_label.setText("0.00")
            if self.agent is not None:
                self.epsilon_label.setText(f"{self.agent.epsilon:.3f}")
                self.action_label.setText("HOLD")
                self.trade_label.setText("-")
            else:
                self.epsilon_label.setText("0.000")
                self.action_label.setText("-")
                self.trade_label.setText("-")
            return

        self.episode_label.setText(str(self.episode))
        self.step_label.setText(str(result.step))
        self.price_label.setText(f"{result.price:.2f}")
        self.position_label.setText(str(result.position))
        self.cash_label.setText(f"{result.cash:.2f}")
        self.equity_label.setText(f"{result.equity:.2f}")
        self.reward_label.setText(f"{result.reward:.2f}")
        self.epsilon_label.setText(f"{result.epsilon:.3f}")
        self.action_label.setText(result.action_name)
        self.trade_label.setText(result.info or "-")

    def _parse_float(self, value: str, field: str, positive: bool = False) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{field} 값이 올바르지 않습니다.") from exc
        if positive and parsed <= 0:
            raise ValueError(f"{field} 값은 0보다 커야 합니다.")
        return parsed

    def _parse_int(self, value: str, field: str) -> int:
        try:
            parsed = int(float(value))
        except ValueError as exc:
            raise ValueError(f"{field} 값이 올바르지 않습니다.") from exc
        if parsed <= 0:
            raise ValueError(f"{field} 값은 1 이상이어야 합니다.")
        return parsed


def main() -> None:
    app = QApplication(sys.argv)
    window = ReinforcementWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
