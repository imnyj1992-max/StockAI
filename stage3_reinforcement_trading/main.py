"""Stage 3: Reinforcement learning driven virtual trading simulator."""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import requests
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SB3_AVAILABLE = False
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore
    gym = None  # type: ignore


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
        base_price: float = 100.0,
        symbol: str = "SYN",
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
        self.base_price = float(base_price)
        self.symbol = symbol
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

        start_price = float(self.rng.normal(self.base_price, max(1.0, self.base_price * 0.02)))
        self.price = max(0.01, start_price)
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
            "symbol": self.symbol,
        }


def _safe_decimal(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return 0.0
    return 0.0


class KiwoomMockClient:
    """Minimal Kiwoom REST helper sufficient for mock account lookups."""

    def __init__(self, host: str = "https://mockapi.kiwoom.com") -> None:
        self.host = host.rstrip("/")
        self._appkey: Optional[str] = None
        self._appsecret: Optional[str] = None
        self._access_token: Optional[str] = None
        self._hashkey_cache: Dict[str, str] = {}

    @property
    def has_token(self) -> bool:
        return bool(self._access_token)

    def set_credentials(self, appkey: str, secret: str) -> None:
        self._appkey = appkey.strip()
        self._appsecret = secret.strip()
        self._hashkey_cache.clear()

    def authenticate(self) -> str:
        if not self._appkey or not self._appsecret:
            raise RuntimeError("모의 App Key와 Secret Key를 입력해주세요.")
        payload = {
            "grant_type": "client_credentials",
            "appkey": self._appkey,
            "secretkey": self._appsecret,
        }
        response = requests.post(
            f"{self.host}/oauth2/token",
            headers={"Content-Type": "application/json;charset=UTF-8"},
            json=payload,
            timeout=10,
        )
        data = self._parse_response(response)
        token = data.get("access_token") or data.get("token")
        if not token:
            raise RuntimeError("응답에 access_token이 없습니다.")
        self._access_token = token
        return token

    def fetch_account(self, payload: Dict[str, Any], *, endpoint: str, tr_id: str, api_id: Optional[str]) -> Dict[str, Any]:
        return self._request(payload, endpoint=endpoint, tr_id=tr_id, api_id=api_id, require_hashkey=False)

    def _request(
        self,
        payload: Dict[str, Any],
        *,
        endpoint: str,
        tr_id: str,
        api_id: Optional[str],
        require_hashkey: bool,
    ) -> Dict[str, Any]:
        if not self._access_token:
            raise RuntimeError("먼저 모의 계좌 인증을 진행해주세요.")
        if not endpoint.startswith("/"):
            raise RuntimeError("Endpoint는 '/'로 시작해야 합니다.")
        payload_json = json.dumps(payload, ensure_ascii=False)
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self._access_token}",
            "appkey": self._appkey or "",
            "appsecret": self._appsecret or "",
            "tr_id": tr_id,
        }
        if api_id:
            headers["api-id"] = api_id
        if require_hashkey:
            hashkey = self._hashkey_cache.get(payload_json)
            if not hashkey:
                hashkey = self._generate_hashkey(payload_json)
                self._hashkey_cache[payload_json] = hashkey
            headers["hashkey"] = hashkey
        response = requests.post(
            f"{self.host}{endpoint}",
            headers=headers,
            data=payload_json.encode("utf-8"),
            timeout=10,
        )
        return self._parse_response(response)

    def _generate_hashkey(self, payload_json: str) -> str:
        response = requests.post(
            f"{self.host}/oauth2/hashkey",
            headers={
                "Content-Type": "application/json;charset=UTF-8",
                "appkey": self._appkey or "",
                "appsecret": self._appsecret or "",
                "authorization": f"Bearer {self._access_token}" if self._access_token else "",
            },
            data=payload_json.encode("utf-8"),
            timeout=10,
        )
        data = self._parse_response(response)
        key = data.get("hashkey") or data.get("HASHKEY")
        if not key:
            raise RuntimeError("Hashkey 응답이 올바르지 않습니다.")
        return key

    @staticmethod
    def _parse_response(response: requests.Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            data = {}
        if response.status_code >= 400:
            message = data.get("message") or data.get("msg") or response.text
            raise RuntimeError(f"HTTP {response.status_code}: {message}")
        if not isinstance(data, dict):
            raise RuntimeError("API 응답이 JSON 객체 형태가 아닙니다.")
        return data

    @staticmethod
    def parse_holdings(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = []
        for key in ("output1", "output2", "stocks", "holdings"):
            value = data.get(key)
            if isinstance(value, list):
                candidates = [item for item in value if isinstance(item, dict)]
                break
        return candidates

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

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialised = {
            "action_count": self.action_count,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "q_table": {
                ",".join(map(str, state)): values.tolist() for state, values in self.q_table.items()
            },
        }
        path.write_text(json.dumps(serialised), encoding="utf-8")

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.action_count = int(data.get("action_count", self.action_count))
        self.alpha = float(data.get("alpha", self.alpha))
        self.gamma = float(data.get("gamma", self.gamma))
        self.epsilon = float(data.get("epsilon", self.epsilon))
        self.epsilon_decay = float(data.get("epsilon_decay", self.epsilon_decay))
        self.min_epsilon = float(data.get("min_epsilon", self.min_epsilon))
        table = defaultdict(lambda: np.zeros(self.action_count, dtype=np.float32))
        for key, values in data.get("q_table", {}).items():
            state = tuple(int(x) for x in key.split(","))  # type: ignore[arg-type]
            table[state] = np.array(values, dtype=np.float32)
        self.q_table = table



if SB3_AVAILABLE:
    class GymTradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, **env_kwargs) -> None:
            self.env_kwargs = env_kwargs
            base_env = TradingEnvironment(**env_kwargs)
            self.core_env = base_env
            obs_len = base_env.window_size + 1
            self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(obs_len,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(len(TradingEnvironment.ACTIONS))

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
            if seed is not None:
                self.core_env.rng = np.random.default_rng(seed)
            state = self.core_env.reset()
            return np.array(state, dtype=np.float32), {}

        def step(self, action: int):  # type: ignore[override]
            next_state, reward, done, info = self.core_env.step(int(action))
            obs = np.array(next_state, dtype=np.float32)
            return obs, reward, done, False, info

        def render(self):  # pragma: no cover - not used
            return None
else:  # pragma: no cover - SB3 unavailable
    GymTradingEnv = None  # type: ignore


class SB3TrainWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, env_kwargs: Dict[str, Any], timesteps: int, save_path: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_kwargs = env_kwargs
        self.timesteps = timesteps
        self.save_path = Path(save_path)

    def run(self) -> None:
        if not SB3_AVAILABLE or GymTradingEnv is None:
            self.failed.emit("SB3 라이브러리가 설치되어 있지 않습니다.")
            return
        try:
            self.progress.emit("[SB3] 학습 환경을 생성합니다.")

            def make_env():
                return GymTradingEnv(**self.env_kwargs)

            vec_env = DummyVecEnv([make_env])
            model = PPO("MlpPolicy", vec_env, verbose=0)
            self.progress.emit(f"[SB3] 총 {self.timesteps:,} 스텝 학습 중 ...")
            model.learn(total_timesteps=self.timesteps)
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(self.save_path))
            self.finished.emit(str(self.save_path))
        except Exception as exc:  # pragma: no cover - runtime error handling
            self.failed.emit(str(exc))



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
        self.kiwoom_client = KiwoomMockClient()
        self.holdings: List[Dict[str, Any]] = []
        self.active_symbol: str = "SYN"
        self.selected_symbol_info: Optional[Dict[str, Any]] = None
        self.sb3_model: Optional[object] = None
        self.sb3_worker: Optional[SB3TrainWorker] = None

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



        account_box = QGroupBox("모의 계좌 연결")

        account_layout = QGridLayout()

        account_box.setLayout(account_layout)

        main_layout.addWidget(account_box)



        self.appkey_input = QLineEdit()

        self.appkey_input.setPlaceholderText("모의 App Key")

        self.secret_input = QLineEdit()

        self.secret_input.setPlaceholderText("모의 Secret Key")

        self.secret_input.setEchoMode(QLineEdit.Password)

        self.account_input = QLineEdit()

        self.account_input.setPlaceholderText("계좌번호 (예: 00000000)")

        self.product_code_input = QLineEdit("01")

        self.custtype_input = QLineEdit("P")

        self.account_endpoint_input = QLineEdit("/api/dostk/acnt")

        self.account_tr_input = QLineEdit("TTTS3004R")

        self.account_api_input = QLineEdit("kt00004")

        self.symbol_input = QLineEdit("005930")

        self.symbol_display = QLabel("활성 종목: -")



        self.connect_button = QPushButton("모의계좌 인증")

        self.connect_button.clicked.connect(self._connect_mock_account)

        self.account_button = QPushButton("계좌 조회")

        self.account_button.clicked.connect(self._fetch_account_snapshot)

        self.symbol_pick_button = QPushButton("활성 종목 선택")

        self.symbol_pick_button.clicked.connect(self._select_active_symbol)



        account_layout.addWidget(QLabel("App Key"), 0, 0)

        account_layout.addWidget(self.appkey_input, 0, 1)

        account_layout.addWidget(QLabel("Secret Key"), 0, 2)

        account_layout.addWidget(self.secret_input, 0, 3)

        account_layout.addWidget(QLabel("계좌번호"), 1, 0)

        account_layout.addWidget(self.account_input, 1, 1)

        account_layout.addWidget(QLabel("상품코드"), 1, 2)

        account_layout.addWidget(self.product_code_input, 1, 3)

        account_layout.addWidget(QLabel("custtype"), 2, 0)

        account_layout.addWidget(self.custtype_input, 2, 1)

        account_layout.addWidget(QLabel("계좌 Endpoint"), 2, 2)

        account_layout.addWidget(self.account_endpoint_input, 2, 3)

        account_layout.addWidget(QLabel("계좌 TR ID"), 3, 0)

        account_layout.addWidget(self.account_tr_input, 3, 1)

        account_layout.addWidget(QLabel("계좌 API ID"), 3, 2)

        account_layout.addWidget(self.account_api_input, 3, 3)

        account_layout.addWidget(QLabel("기본 종목 코드"), 4, 0)

        account_layout.addWidget(self.symbol_input, 4, 1)

        account_layout.addWidget(self.symbol_display, 4, 2, 1, 2)

        account_layout.addWidget(self.connect_button, 5, 0, 1, 2)

        account_layout.addWidget(self.account_button, 5, 2, 1, 2)

        account_layout.addWidget(self.symbol_pick_button, 6, 0, 1, 4)
        self.account_status_label = QLabel("계좌 정보: -")
        self.holdings_status_label = QLabel("보유 종목: -")
        account_layout.addWidget(self.account_status_label, 7, 0, 1, 4)
        account_layout.addWidget(self.holdings_status_label, 8, 0, 1, 4)

        model_box = QGroupBox("모델 관리")

        model_layout = QGridLayout()

        model_box.setLayout(model_layout)

        main_layout.addWidget(model_box)



        self.agent_mode_combo = QComboBox()
        self.agent_mode_combo.addItems(["Q-Learning", "SB3-PPO"])
        self.agent_mode_combo.currentTextChanged.connect(self._on_agent_mode_changed)

        self.model_path_default = Path(__file__).with_name("stage3_q_table.json")
        self.model_path_input = QLineEdit(str(self.model_path_default))
        self.load_model_button = QPushButton("Q 모델 불러오기")
        self.save_model_button = QPushButton("Q 모델 저장")
        self.load_model_button.clicked.connect(self._manual_load_model)
        self.save_model_button.clicked.connect(self._manual_save_model)

        self.sb3_model_path_default = Path(__file__).with_name("stage3_sb3.zip")
        self.sb3_model_path_input = QLineEdit(str(self.sb3_model_path_default))
        self.sb3_timesteps_input = QLineEdit("50000")
        self.sb3_train_button = QPushButton("SB3 학습")
        self.sb3_train_button.setEnabled(SB3_AVAILABLE)
        if not SB3_AVAILABLE:
            self.sb3_train_button.setToolTip("stable-baselines3 미설치")
        self.sb3_train_button.clicked.connect(self._train_sb3_model)
        self.sb3_load_button = QPushButton("SB3 모델 불러오기")
        self.sb3_load_button.clicked.connect(self._load_sb3_model_from_button)
        self.sb3_status_label = QLabel("SB3 모델: -")

        model_layout.addWidget(QLabel("에이전트"), 0, 0)
        model_layout.addWidget(self.agent_mode_combo, 0, 1, 1, 2)
        model_layout.addWidget(QLabel("Q 모델 경로"), 1, 0)
        model_layout.addWidget(self.model_path_input, 1, 1, 1, 2)
        model_layout.addWidget(self.load_model_button, 2, 0)
        model_layout.addWidget(self.save_model_button, 2, 1)
        model_layout.addWidget(QLabel("SB3 모델 경로"), 3, 0)
        model_layout.addWidget(self.sb3_model_path_input, 3, 1, 1, 2)
        model_layout.addWidget(QLabel("SB3 스텝"), 4, 0)
        model_layout.addWidget(self.sb3_timesteps_input, 4, 1)
        model_layout.addWidget(self.sb3_train_button, 4, 2)
        model_layout.addWidget(self.sb3_load_button, 5, 0)
        model_layout.addWidget(self.sb3_status_label, 5, 1, 1, 2)






        config_box = QGroupBox("강화학습 설정")

        config_layout = QGridLayout()

        config_box.setLayout(config_layout)

        main_layout.addWidget(config_box)



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

            ("에피소드 길이", self.episode_input),

            ("상태 창크기", self.window_input),

            ("학습률", self.alpha_input),

            ("감가율", self.gamma_input),

            ("탐험률", self.epsilon_input),

            ("감소율", self.decay_input),

            ("최소 탐험률", self.min_epsilon_input),

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
        self.symbol_status_label = QLabel("-")

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
            ("종목", self.symbol_status_label),
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
        mode = self.agent_mode_combo.currentText()
        need_init = self.env is None or self.current_state is None or (mode == "Q-Learning" and self.agent is None)
        if need_init:
            try:
                self._initialise_environment()
            except ValueError as exc:
                QMessageBox.critical(self, "설정 오류", str(exc))
                return
        if mode == "SB3-PPO" and (not SB3_AVAILABLE or self.sb3_model is None):
            QMessageBox.warning(self, "SB3 모델", "SB3 모델을 먼저 학습/불러오세요.")
            return
        if not self.timer.isActive():
            self.timer.start()
            self._append_log("학습을 시작합니다.")

    def pause_training(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self._append_log("학습을 일시정지하였습니다.")
        self._save_model()
        self._save_sb3_model()

    def reset_training(self) -> None:
        self.timer.stop()
        self._save_model()
        self._save_sb3_model()
        try:
            self._initialise_environment()
        except ValueError as exc:
            QMessageBox.critical(self, "설정 오류", str(exc))
            return
        self.log_view.clear()
        self._append_log("환경과 에이전트를 리셋하였습니다.")

    def _initialise_environment(self) -> None:
        env_kwargs = self._sb3_env_kwargs()
        symbol = env_kwargs["symbol"]
        self.symbol_display.setText(f"활성 종목: {symbol}")
        self.env = TradingEnvironment(**env_kwargs)
        mode = self.agent_mode_combo.currentText()
        if mode == "Q-Learning":
            alpha = self._parse_float(self.alpha_input.text(), "학습률", positive=True)
            gamma = self._parse_float(self.gamma_input.text(), "감가율", positive=True)
            epsilon = self._parse_float(self.epsilon_input.text(), "탐험률", positive=True)
            decay = self._parse_float(self.decay_input.text(), "감소율", positive=True)
            min_epsilon = self._parse_float(self.min_epsilon_input.text(), "최소 탐험률", positive=True)
            self.agent = QLearningAgent(
                action_count=len(TradingEnvironment.ACTIONS),
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_decay=decay,
                min_epsilon=min_epsilon,
            )
            self._load_model_if_available()
        else:
            self.agent = None
            if not self.sb3_model:
                self._append_log("[SB3] 학습된 모델이 없습니다. SB3 학습 버튼을 사용하세요.")
        self.current_state = self.env.reset()
        self.episode = 1
        self._update_status_labels(reset=True)

    def _on_timer(self) -> None:
        if self.env is None or self.current_state is None:
            self.timer.stop()
            return

        mode = self.agent_mode_combo.currentText()
        if mode == "SB3-PPO":
            if not SB3_AVAILABLE or self.sb3_model is None:
                self.timer.stop()
                QMessageBox.warning(self, "SB3 모델", "SB3 모델을 먼저 학습/불러오세요.")
                return
            obs = np.array(self.current_state, dtype=np.float32)
            action, _ = self.sb3_model.predict(obs, deterministic=False)
            action = int(action)
        else:
            if self.agent is None:
                self.timer.stop()
                return
            action = self.agent.select_action(self.current_state)
        next_state, reward, done, info = self.env.step(action)
        if self.agent is not None and mode == "Q-Learning":
            self.agent.update(self.current_state, action, reward, next_state, done)

        metrics = self.env.metrics()
        epsilon_value = self.agent.epsilon if (self.agent is not None and mode == "Q-Learning") else 0.0
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
                epsilon=epsilon_value,
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
        self._append_log(log_line)

        self.current_state = next_state

        if done:
            episode_reward = self.env.total_reward
            if mode == "Q-Learning" and self.agent is not None:
                epsilon_text = f"ε={self.agent.epsilon:.3f}"
            else:
                epsilon_text = "SB3"
            self._append_log(
                f"에피소드 {self.episode} 종료 - 총 보상 {episode_reward:.2f}, {epsilon_text}"
            )
            if mode == "Q-Learning" and self.agent is not None:
                self.agent.on_episode_end()
                self._save_model()
            else:
                self._save_sb3_model()
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
                self.epsilon_label.setText("-")
                self.action_label.setText("-")
                self.trade_label.setText("-")
            if self.env is not None:
                self.symbol_status_label.setText(self.env.symbol)
            else:
                self.symbol_status_label.setText(self.active_symbol)
            return

        self.episode_label.setText(str(self.episode))
        self.step_label.setText(str(result.step))
        self.price_label.setText(f"{result.price:.2f}")
        self.position_label.setText(str(result.position))
        self.cash_label.setText(f"{result.cash:.2f}")
        self.equity_label.setText(f"{result.equity:.2f}")
        self.reward_label.setText(f"{result.reward:.2f}")
        if self.agent_mode_combo.currentText() == "Q-Learning" and self.agent is not None:
            self.epsilon_label.setText(f"{result.epsilon:.3f}")
        else:
            self.epsilon_label.setText("-")
        self.action_label.setText(result.action_name)
        self.trade_label.setText(result.info or "-")
        symbol_name = getattr(self.env, 'symbol', self.active_symbol)
        self.symbol_status_label.setText(symbol_name)

    def _append_log(self, message: str) -> None:
        self.log_view.append(message)
        self.log_view.ensureCursorVisible()

    def _on_agent_mode_changed(self, mode: str) -> None:
        if mode == "SB3-PPO" and not SB3_AVAILABLE:
            QMessageBox.information(self, "SB3 사용 불가", "stable-baselines3가 설치되어 있지 않습니다. Q-Learning으로 전환합니다.")
            self.agent_mode_combo.blockSignals(True)
            self.agent_mode_combo.setCurrentText("Q-Learning")
            self.agent_mode_combo.blockSignals(False)
            mode = "Q-Learning"
        if mode == "SB3-PPO" and not self.sb3_model:
            self._append_log("[SB3] 학습된 모델이 없습니다. SB3 학습 버튼을 사용하세요.")
        self._update_status_labels(reset=True)

    # ------------------------------------------------------------------
    # Mock account helpers
    # ------------------------------------------------------------------
    def _connect_mock_account(self) -> None:
        appkey = self.appkey_input.text().strip()
        secret = self.secret_input.text().strip()
        if not appkey or not secret:
            QMessageBox.warning(self, "입력 필요", "모의 App Key와 Secret Key를 입력해주세요.")
            return
        self.kiwoom_client.set_credentials(appkey, secret)
        try:
            self.kiwoom_client.authenticate()
        except Exception as exc:
            QMessageBox.critical(self, "모의계좌 인증 실패", str(exc))
            self._append_log(f"[Account] 인증 실패: {exc}")
            return
        QMessageBox.information(self, "모의계좌 인증", "액세스 토큰을 발급받았습니다.")
        self._append_log("[Account] 인증 완료")

    def _fetch_account_snapshot(self) -> None:
        if not self.kiwoom_client.has_token:
            QMessageBox.warning(self, "모의계좌 인증", "먼저 모의계좌 인증을 진행해주세요.")
            return
        try:
            payload = self._build_account_payload()
        except ValueError as exc:
            QMessageBox.warning(self, "입력 필요", str(exc))
            return
        endpoint = self.account_endpoint_input.text().strip() or "/api/dostk/acnt"
        tr_id = self.account_tr_input.text().strip() or "TTTS3004R"
        api_id = self.account_api_input.text().strip() or None
        try:
            response = self.kiwoom_client.fetch_account(payload, endpoint=endpoint, tr_id=tr_id, api_id=api_id)
        except Exception as exc:
            QMessageBox.critical(self, "계좌 조회 실패", str(exc))
            self._append_log(f"[Account] 조회 실패: {exc}")
            return
        summary_text = self._format_account_summary(response)
        holdings = KiwoomMockClient.parse_holdings(response)
        self.holdings = holdings
        self.account_status_label.setText(summary_text)
        holdings_names = ", ".join(filter(None, (self._extract_symbol(item) for item in holdings[:5])))
        if not holdings_names:
            holdings_names = "-"
        self.holdings_status_label.setText(f"보유 종목 {len(holdings)}건: {holdings_names}")
        self._append_log(f"[Account] {summary_text}")
        self._append_log(f"[Account] Holdings: {holdings_names}")
        if holdings:
            self.selected_symbol_info = holdings[0]
            symbol = self._extract_symbol(holdings[0])
            if symbol:
                self.active_symbol = symbol
                self.symbol_display.setText(f"활성 종목: {self.active_symbol}")

    def _select_active_symbol(self) -> None:
        symbol = self.symbol_input.text().strip()
        entry = None
        if not symbol and self.holdings:
            entry = max(self.holdings, key=self._holding_score, default=None)
            if entry:
                symbol = self._extract_symbol(entry) or symbol
        if not symbol:
            QMessageBox.warning(self, "종목 선택", "종목 코드를 직접 입력하거나 계좌 조회 후 보유 종목을 선택해주세요.")
            return
        self.active_symbol = symbol.upper()
        self.symbol_display.setText(f"활성 종목: {self.active_symbol}")
        if entry is None:
            entry = self._find_holding(self.active_symbol)
        self.selected_symbol_info = entry
        if entry:
            self._append_log(f"[AI] 활성 종목 설정: {self._holding_description(entry)}")
        else:
            self._append_log(f"[AI] 활성 종목 설정: {self.active_symbol}")

    def _build_account_payload(self) -> Dict[str, Any]:
        account = self.account_input.text().strip()
        if not account:
            raise ValueError("계좌번호를 입력해주세요.")
        product = self.product_code_input.text().strip() or "01"
        return {
            "CANO": account,
            "ACNT_PRDT_CD": product,
            "qry_tp": "0",
            "dmst_stex_tp": "KRX",
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "N",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "FUND_STTL_DVSN_CD": "01",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

    def _format_account_summary(self, payload: Dict[str, Any]) -> str:
        summary = payload.get("output") or payload.get("output1") or payload
        d2 = _safe_decimal(summary.get("d2_entra_amt") or summary.get("d2_deposit"))
        eval_amt = _safe_decimal(summary.get("tot_evlu_amt") or summary.get("eval_amt"))
        cash = _safe_decimal(summary.get("prsm_dpst") or summary.get("prsnl_cash") or summary.get("prsm_dpst_aset_amt"))
        pnl = _safe_decimal(summary.get("tot_pnl_amt") or summary.get("lspft_amt"))
        return f"계좌 정보: 예수금 {d2:,.0f}원 / 평가금 {eval_amt:,.0f}원 / 현금 {cash:,.0f}원 / 손익 {pnl:,.0f}원"

    def _extract_symbol(self, entry: Dict[str, Any]) -> Optional[str]:
        for key in ("pdno", "stk_cd", "stock_code", "symbol", "code"):
            value = entry.get(key)
            if value:
                return str(value).strip()
        return None

    def _find_holding(self, symbol: str) -> Optional[Dict[str, Any]]:
        for entry in self.holdings:
            if self._extract_symbol(entry) == symbol:
                return entry
        return None

    def _holding_score(self, entry: Dict[str, Any]) -> float:
        qty = _safe_decimal(entry.get("hldg_qty") or entry.get("ord_psbl_qty") or entry.get("qty"))
        value = _safe_decimal(entry.get("evlt_amt") or entry.get("eval_amt") or entry.get("pchs_amt"))
        return qty * max(value, 1.0)

    def _holding_description(self, entry: Dict[str, Any]) -> str:
        symbol = self._extract_symbol(entry) or "-"
        qty = _safe_decimal(entry.get("hldg_qty") or entry.get("ord_psbl_qty") or entry.get("qty"))
        price = _safe_decimal(entry.get("pchs_avg_prc") or entry.get("avg_prc") or entry.get("price"))
        return f"{symbol} (보유 {qty:.0f}주, 평균 {price:,.0f}원)"

    # ------------------------------------------------------------------
    # Model helpers
    def _train_sb3_model(self) -> None:
        if not SB3_AVAILABLE:
            QMessageBox.warning(self, "SB3 학습", "stable-baselines3가 설치되어 있지 않습니다.")
            return
        if self.sb3_worker and self.sb3_worker.isRunning():
            QMessageBox.information(self, "SB3 학습", "이미 학습이 진행 중입니다.")
            return
        try:
            timesteps = self._parse_int(self.sb3_timesteps_input.text(), "SB3 스텝")
        except ValueError as exc:
            QMessageBox.warning(self, "SB3 학습", str(exc))
            return
        try:
            env_kwargs = self._sb3_env_kwargs()
        except ValueError as exc:
            QMessageBox.warning(self, "SB3 학습", str(exc))
            return
        save_path = self._sb3_model_path()
        self.sb3_train_button.setEnabled(False)
        self.sb3_train_button.setText("SB3 학습 중...")
        self.sb3_worker = SB3TrainWorker(env_kwargs, timesteps, save_path, parent=self)
        self.sb3_worker.progress.connect(self._append_log)
        self.sb3_worker.finished.connect(self._on_sb3_training_finished)
        self.sb3_worker.failed.connect(self._on_sb3_training_failed)
        self.sb3_worker.start()

    def _on_sb3_training_finished(self, path_str: str) -> None:
        self.sb3_train_button.setEnabled(True)
        self.sb3_train_button.setText("SB3 학습")
        self._append_log(f"[SB3] 학습 완료: {path_str}")
        self.sb3_worker = None
        self._load_sb3_model(Path(path_str))

    def _on_sb3_training_failed(self, message: str) -> None:
        self.sb3_train_button.setEnabled(True)
        self.sb3_train_button.setText("SB3 학습")
        self._append_log(f"[SB3] 학습 실패: {message}")
        QMessageBox.critical(self, "SB3 학습 실패", message)
        self.sb3_worker = None

    def _load_sb3_model_from_button(self) -> None:
        path = self._sb3_model_path()
        if not path.exists():
            QMessageBox.warning(self, "SB3 모델", "모델 파일이 존재하지 않습니다.")
            return
        self._load_sb3_model(path)

    def _load_sb3_model(self, path: Path) -> None:
        if not SB3_AVAILABLE:
            QMessageBox.warning(self, "SB3 모델", "stable-baselines3가 설치되어 있지 않습니다.")
            return
        try:
            self.sb3_model = PPO.load(str(path))
            self.sb3_status_label.setText(f"SB3 모델: {path.name}")
            self._append_log(f"[SB3] 모델 불러오기: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "SB3 모델 불러오기 실패", str(exc))

    # ------------------------------------------------------------------
    def _model_path(self) -> Path:
        raw = self.model_path_input.text().strip()
        if not raw:
            return self.model_path_default
        return Path(raw)

    def _sb3_model_path(self) -> Path:
        raw = self.sb3_model_path_input.text().strip()
        if not raw:
            return self.sb3_model_path_default
        return Path(raw)

    def _sb3_env_kwargs(self) -> Dict[str, Any]:
        initial_cash = self._parse_float(self.cash_input.text(), "초기 현금")
        episode_steps = self._parse_int(self.episode_input.text(), "에피소드 길이")
        window_size = self._parse_int(self.window_input.text(), "상태 창크기")
        symbol = self.active_symbol or self.symbol_input.text().strip() or "SYN"
        base_price, volatility = self._derive_market_profile(symbol)
        return {
            "initial_cash": initial_cash,
            "episode_length": episode_steps,
            "window_size": window_size,
            "volatility": volatility,
            "symbol": symbol,
            "base_price": base_price,
        }

    def _manual_load_model(self) -> None:
        if self.agent_mode_combo.currentText() == "SB3-PPO":
            self._load_sb3_model_from_button()
            return
        if self.agent is None:
            try:
                self._initialise_environment()
            except ValueError as exc:
                QMessageBox.critical(self, "환경 초기화 실패", str(exc))
                return
        try:
            self._load_model_from_path(self._model_path())
        except FileNotFoundError:
            QMessageBox.warning(self, "모델 불러오기", "모델 파일이 존재하지 않습니다.")
        except Exception as exc:
            QMessageBox.critical(self, "모델 불러오기 실패", str(exc))

    def _manual_save_model(self) -> None:
        if self.agent_mode_combo.currentText() == "SB3-PPO":
            if not SB3_AVAILABLE:
                QMessageBox.warning(self, "모델 저장", "SB3 라이브러리가 없어 저장할 수 없습니다.")
                return
            if not self.sb3_model:
                QMessageBox.warning(self, "모델 저장", "저장할 SB3 모델이 없습니다.")
                return
            self._save_sb3_model()
            return
        if not self.agent:
            QMessageBox.warning(self, "모델 저장", "저장할 에이전트가 없습니다.")
            return
        self._save_model()

    def _save_model(self) -> None:
        if not self.agent:
            return
        path = self._model_path()
        try:
            self.agent.save(path)
            self._append_log(f"[Model] 저장: {path}")
        except Exception as exc:
            self._append_log(f"[Model] 저장 실패: {exc}")

    def _save_sb3_model(self) -> None:
        if not SB3_AVAILABLE or self.sb3_model is None:
            return
        path = self._sb3_model_path()
        try:
            self.sb3_model.save(str(path))
            self._append_log(f"[SB3] 모델 저장: {path}")
        except Exception as exc:
            self._append_log(f"[SB3] 모델 저장 실패: {exc}")

    def _load_model_from_path(self, path: Path) -> None:
        if not self.agent:
            raise RuntimeError("Q-Learning agent is not initialised.")
        self.agent.load(path)
        self._append_log(f"[Model] 불러오기: {path}")

    def _load_model_if_available(self) -> None:
        path = self._model_path()
        if path.exists() and self.agent is not None:
            try:
                self._load_model_from_path(path)
            except Exception as exc:
                self._append_log(f"[Model] 불러오기 실패: {exc}")

    def _derive_market_profile(self, symbol: str) -> Tuple[float, float]:
        entry = self._find_holding(symbol)
        if not entry:
            return 100.0, 0.01
        avg_price = _safe_decimal(entry.get("pchs_avg_prc") or entry.get("avg_prc") or entry.get("price"))
        eval_amt = _safe_decimal(entry.get("evlt_amt") or entry.get("eval_amt"))
        qty = _safe_decimal(entry.get("hldg_qty") or entry.get("ord_psbl_qty") or entry.get("qty"))
        base_price = avg_price if avg_price > 0 else max(10.0, eval_amt / max(qty, 1.0))
        volatility = min(0.05, 0.005 + (qty / max(1.0, eval_amt / 1_000_000)))
        return base_price, max(0.001, volatility)

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



    def closeEvent(self, event) -> None:
        self._save_model()
        self._save_sb3_model()
        super().closeEvent(event)

def main() -> None:
    app = QApplication(sys.argv)
    window = ReinforcementWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
