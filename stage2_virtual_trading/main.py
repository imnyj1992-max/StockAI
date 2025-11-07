"""Stage 2: Virtual trading simulator GUI.

This module builds on top of the project scaffolding from stage 1 but
replaces the Kiwoom OpenAPI dependency with a fully self-contained virtual
broker.  The simulator allows the user to initialise a cash balance and then
perform buy/sell/hold actions to validate the GUI flow before connecting to
real trading infrastructure.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from ai import (
        MockTrainingConfig,
        SelectorConfig,
        WalkForwardConfig,
        generate_synthetic_universe,
        make_vec_envs,
        prepare_datasets,
        run_selector_stage,
        train_rl_agent,
    )

    AI_MODULE_AVAILABLE = True
except Exception:  # pragma: no cover - allow running without package install
    try:
        from stage2_virtual_trading.ai import (
            MockTrainingConfig,
            SelectorConfig,
            WalkForwardConfig,
            generate_synthetic_universe,
            make_vec_envs,
            prepare_datasets,
            run_selector_stage,
            train_rl_agent,
        )

        AI_MODULE_AVAILABLE = True
    except Exception:  # pragma: no cover
        AI_MODULE_AVAILABLE = False


@dataclass
class Position:
    """Represents a holding for a single symbol."""

    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    last_price: float = 0.0

    def update_on_buy(self, price: float, quantity: int) -> None:
        """Update the position after a purchase."""

        if quantity <= 0:
            raise ValueError("매수 수량은 1 이상이어야 합니다.")
        total_cost = self.avg_price * self.quantity + price * quantity
        self.quantity += quantity
        self.avg_price = total_cost / self.quantity
        self.last_price = price

    def update_on_sell(self, price: float, quantity: int) -> None:
        """Update the position after a sale."""

        if quantity <= 0:
            raise ValueError("매도 수량은 1 이상이어야 합니다.")
        if quantity > self.quantity:
            raise ValueError("보유 수량보다 많은 수량을 매도할 수 없습니다.")
        self.quantity -= quantity
        self.last_price = price
        if self.quantity == 0:
            self.avg_price = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealised_pnl(self) -> float:
        return self.quantity * (self.last_price - self.avg_price)


class VirtualBroker:
    """Simple broker that tracks cash, positions, and P&L."""

    def __init__(self, initial_cash: float) -> None:
        if initial_cash <= 0:
            raise ValueError("초기 현금은 0보다 커야 합니다.")
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.realised_pnl = 0.0
        self.history: List[str] = []

    # ------------------------------------------------------------------
    # Trading operations
    # ------------------------------------------------------------------
    def buy(self, symbol: str, price: float, quantity: int) -> None:
        symbol = symbol.upper()
        if price <= 0:
            raise ValueError("가격은 0보다 커야 합니다.")
        if quantity <= 0:
            raise ValueError("수량은 1 이상이어야 합니다.")
        cost = price * quantity
        if cost > self.cash:
            raise ValueError("현금 잔액이 부족합니다.")

        position = self.positions.setdefault(symbol, Position(symbol))
        position.update_on_buy(price, quantity)
        self.cash -= cost
        self.history.append(f"BUY {symbol} x{quantity} @ {price:.2f}")

    def sell(self, symbol: str, price: float, quantity: int) -> None:
        symbol = symbol.upper()
        if symbol not in self.positions or self.positions[symbol].quantity == 0:
            raise ValueError("보유 중인 종목이 아닙니다.")
        if price <= 0:
            raise ValueError("가격은 0보다 커야 합니다.")
        if quantity <= 0:
            raise ValueError("수량은 1 이상이어야 합니다.")

        position = self.positions[symbol]
        avg_price = position.avg_price
        position.update_on_sell(price, quantity)
        proceeds = price * quantity
        self.cash += proceeds
        realised = (price - avg_price) * quantity
        self.realised_pnl += realised
        self.history.append(f"SELL {symbol} x{quantity} @ {price:.2f}")

        if position.quantity == 0:
            # Remove empty positions to keep the table clean.
            del self.positions[symbol]

    def hold(self, symbol: str) -> None:
        symbol = symbol.upper()
        if symbol:
            self.history.append(f"HOLD {symbol}")
        else:
            self.history.append("HOLD")

    # ------------------------------------------------------------------
    # Portfolio metrics
    # ------------------------------------------------------------------
    @property
    def market_value(self) -> float:
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    @property
    def total_pnl(self) -> float:
        return self.realised_pnl + sum(pos.unrealised_pnl for pos in self.positions.values())


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


class KiwoomOrderClient:
    """Minimal REST client for Kiwoom mock-trading order APIs."""

    def __init__(self, host: str = "https://mockapi.kiwoom.com") -> None:
        self.host = host.rstrip("/")
        self._appkey: Optional[str] = None
        self._appsecret: Optional[str] = None
        self._access_token: Optional[str] = None
        self._hashkey_cache: Dict[str, str] = {}

    @property
    def access_token(self) -> Optional[str]:
        return self._access_token

    def set_credentials(self, appkey: str, secret: str) -> None:
        self._appkey = appkey.strip()
        self._appsecret = secret.strip()
        self._hashkey_cache.clear()

    def authenticate(self) -> str:
        if not self._appkey or not self._appsecret:
            raise RuntimeError("모의 App Key와 Secret Key를 먼저 입력해주세요.")
        url = f"{self.host}/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self._appkey,
            "secretkey": self._appsecret,
        }
        headers = {"Content-Type": "application/json;charset=UTF-8"}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
        except requests.RequestException as exc:  # pragma: no cover - network
            raise RuntimeError(f"토큰 발급 요청 실패: {exc}") from exc

        data = self._parse_response(response)
        token = data.get("access_token") or data.get("token") or data.get("accessToken")
        if not token:
            raise RuntimeError("응답에서 access_token을 찾을 수 없습니다.")
        self._access_token = token
        return token

    def place_order(
        self,
        *,
        endpoint: str,
        tr_id: str,
        custtype: str,
        payload: Dict[str, Any],
        api_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._request(
            endpoint=endpoint,
            tr_id=tr_id,
            custtype=custtype,
            payload=payload,
            api_id=api_id,
            require_hashkey=True,
        )

    def fetch_account_summary(
        self,
        *,
        endpoint: str,
        tr_id: str,
        custtype: str,
        payload: Dict[str, Any],
        api_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._request(
            endpoint=endpoint,
            tr_id=tr_id,
            custtype=custtype,
            payload=payload,
            api_id=api_id,
            require_hashkey=False,
        )

    def _request(
        self,
        *,
        endpoint: str,
        tr_id: str,
        custtype: str,
        payload: Dict[str, Any],
        api_id: Optional[str],
        require_hashkey: bool,
    ) -> Dict[str, Any]:
        if not self._access_token:
            raise RuntimeError("모의투자 인증을 먼저 진행해주세요.")
        if not self._appkey or not self._appsecret:
            raise RuntimeError("App Key/Secret이 설정되지 않았습니다.")
        if not endpoint.startswith("/"):
            raise RuntimeError("Endpoint는 '/'로 시작해야 합니다. (예: /api/dostk/v1/order)")

        payload_json = json.dumps(payload, ensure_ascii=False)
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self._access_token}",
            "appkey": self._appkey,
            "appsecret": self._appsecret,
            "custtype": custtype or "P",
            "tr_id": tr_id,
        }
        if api_id:
            headers["api-id"] = api_id

        if require_hashkey:
            hashkey = self._generate_hashkey(payload_json)
            if hashkey:
                headers["hashkey"] = hashkey

        url = f"{self.host}{endpoint}"
        try:
            response = requests.post(url, headers=headers, data=payload_json.encode("utf-8"), timeout=10)
        except requests.RequestException as exc:  # pragma: no cover - network
            raise RuntimeError(f"HTTP 요청 실패: {exc}") from exc

        return self._parse_response(response)

    def _generate_hashkey(self, payload_json: str) -> Optional[str]:
        cache_hit = self._hashkey_cache.get(payload_json)
        if cache_hit:
            return cache_hit

        url = f"{self.host}/oauth2/hashkey"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "appkey": self._appkey or "",
            "appsecret": self._appsecret or "",
            "authorization": f"Bearer {self._access_token}" if self._access_token else "",
        }
        try:
            response = requests.post(url, headers=headers, data=payload_json.encode("utf-8"), timeout=10)
        except requests.RequestException as exc:  # pragma: no cover - network
            raise RuntimeError(f"Hashkey 발급 실패: {exc}") from exc

        data = self._parse_response(response)
        key = data.get("HASHKEY") or data.get("hashkey")
        if not key:
            raise RuntimeError("Hashkey 응답에 hashkey 필드가 없습니다.")
        self._hashkey_cache[payload_json] = key
        return key

    @staticmethod
    def _parse_response(response: requests.Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            data = {}
        if response.status_code >= 400:
            message = data.get("msg") or data.get("message") or response.text
            raise RuntimeError(f"HTTP {response.status_code}: {message}")
        if not isinstance(data, dict):
            raise RuntimeError("API 응답이 JSON 객체 형태가 아닙니다.")
        return data


class VirtualTradingWindow(QMainWindow):
    """Main GUI window for stage 2 virtual trading."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StockAI - Stage 2: Virtual Trading Simulator")
        self.resize(720, 520)

        self.broker: Optional[VirtualBroker] = None
        self.order_client: Optional[KiwoomOrderClient] = None

        self.appkey_input = QLineEdit()
        self.appkey_input.setPlaceholderText("모의 App Key")
        self.secret_input = QLineEdit()
        self.secret_input.setPlaceholderText("모의 Secret Key")
        self.secret_input.setEchoMode(QLineEdit.Password)
        self.account_input = QLineEdit()
        self.account_input.setPlaceholderText("계좌번호 (예: 00000000)")
        self.product_code_input = QLineEdit("01")
        self.custtype_input = QLineEdit("P")
        self.order_division_input = QLineEdit("00")
        self.order_endpoint_input = QLineEdit("/api/dostk/v1/order")
        self.order_api_id_input = QLineEdit()
        self.order_api_id_input.setPlaceholderText("주문 api-id (optional)")
        self.buy_tr_id_input = QLineEdit("VTTC0802U")
        self.sell_tr_id_input = QLineEdit("VTTC0801U")
        self.balance_endpoint_input = QLineEdit("/api/dostk/acnt")
        self.balance_tr_id_input = QLineEdit("TTTS3004R")
        self.balance_api_id_input = QLineEdit("kt00004")
        self.token_button = QPushButton("모의투자 인증")
        self.token_button.clicked.connect(self._authenticate_mock)
        self.fetch_balance_button = QPushButton("모의 계좌 조회")
        self.fetch_balance_button.clicked.connect(self._fetch_account_snapshot)

        # ------------------------------------------------------------------
        # Widgets
        # ------------------------------------------------------------------
        self.initial_cash_input = QLineEdit("1000000")
        self.start_button = QPushButton("가상 거래 시작")
        self.start_button.clicked.connect(self._initialise_broker)

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("종목 코드")

        self.price_input = QLineEdit()
        self.price_input.setPlaceholderText("가격")

        self.quantity_input = QLineEdit()
        self.quantity_input.setPlaceholderText("수량")

        self.buy_button = QPushButton("매수")
        self.buy_button.clicked.connect(self._buy)
        self.sell_button = QPushButton("매도")
        self.sell_button.clicked.connect(self._sell)
        self.hold_button = QPushButton("관망")
        self.hold_button.clicked.connect(self._hold)
        self._set_trade_controls_enabled(False)
        self.auto_train_button = QPushButton("Auto Learn (Selector + RL)")
        self.auto_train_button.clicked.connect(self._trigger_auto_training)
        self.auto_train_button.setEnabled(AI_MODULE_AVAILABLE)

        self.balance_label = QLabel("현금: - / 평가금액: - / 손익: -")
        self.account_snapshot_label = QLabel("계좌 정보: -")
        self.ai_status_label = QLabel("AI 상태: 대기")
        self.auto_trainer = None

        self.positions_table = QTableWidget(0, 6)
        self.positions_table.setHorizontalHeaderLabels(
            ["종목", "수량", "평균 단가", "현재가", "평가금액", "평가손익"]
        )
        self.positions_table.horizontalHeader().setStretchLastSection(True)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        # Layout setup
        settings_layout = QGridLayout()
        settings_layout.addWidget(QLabel("모의 App Key"), 0, 0)
        settings_layout.addWidget(self.appkey_input, 0, 1)
        settings_layout.addWidget(QLabel("모의 Secret Key"), 0, 2)
        settings_layout.addWidget(self.secret_input, 0, 3)
        settings_layout.addWidget(QLabel("계좌번호"), 1, 0)
        settings_layout.addWidget(self.account_input, 1, 1)
        settings_layout.addWidget(QLabel("상품 코드"), 1, 2)
        settings_layout.addWidget(self.product_code_input, 1, 3)
        settings_layout.addWidget(QLabel("Custtype"), 2, 0)
        settings_layout.addWidget(self.custtype_input, 2, 1)
        settings_layout.addWidget(QLabel("주문 구분 (ORD_DVSN)"), 2, 2)
        settings_layout.addWidget(self.order_division_input, 2, 3)
        settings_layout.addWidget(QLabel("주문 Endpoint"), 3, 0)
        settings_layout.addWidget(self.order_endpoint_input, 3, 1, 1, 3)
        settings_layout.addWidget(QLabel("주문 API ID"), 4, 0)
        settings_layout.addWidget(self.order_api_id_input, 4, 1)
        settings_layout.addWidget(QLabel("매수 TR ID"), 4, 2)
        settings_layout.addWidget(self.buy_tr_id_input, 4, 3)
        settings_layout.addWidget(QLabel("매도 TR ID"), 5, 0)
        settings_layout.addWidget(self.sell_tr_id_input, 5, 1)
        settings_layout.addWidget(QLabel("계좌 Endpoint"), 5, 2)
        settings_layout.addWidget(self.balance_endpoint_input, 5, 3)
        settings_layout.addWidget(QLabel("계좌 TR ID"), 6, 0)
        settings_layout.addWidget(self.balance_tr_id_input, 6, 1)
        settings_layout.addWidget(QLabel("계좌 API ID"), 6, 2)
        settings_layout.addWidget(self.balance_api_id_input, 6, 3)
        settings_layout.addWidget(self.token_button, 7, 0, 1, 2, alignment=Qt.AlignRight)
        settings_layout.addWidget(self.fetch_balance_button, 7, 2, 1, 2, alignment=Qt.AlignRight)
        settings_box = QGroupBox("모의투자 설정")
        settings_box.setLayout(settings_layout)

        control_layout = QGridLayout()
        control_layout.addWidget(QLabel("초기 현금"), 0, 0)
        control_layout.addWidget(self.initial_cash_input, 0, 1)
        control_layout.addWidget(self.start_button, 0, 2)
        control_layout.addWidget(self.auto_train_button, 0, 3)

        trade_layout = QHBoxLayout()
        trade_layout.addWidget(self.symbol_input)
        trade_layout.addWidget(self.price_input)
        trade_layout.addWidget(self.quantity_input)
        trade_layout.addWidget(self.buy_button)
        trade_layout.addWidget(self.sell_button)
        trade_layout.addWidget(self.hold_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(settings_box)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(trade_layout)
        main_layout.addWidget(self.account_snapshot_label)
        main_layout.addWidget(self.balance_label)
        main_layout.addWidget(self.ai_status_label)
        main_layout.addWidget(self.positions_table)
        main_layout.addWidget(QLabel("거래 로그"))
        main_layout.addWidget(self.log_output)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _initialise_broker(self) -> None:
        try:
            initial_cash = float(self.initial_cash_input.text())
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "초기 현금은 숫자여야 합니다.")
            return

        try:
            self.broker = VirtualBroker(initial_cash)
        except ValueError as exc:
            QMessageBox.warning(self, "입력 오류", str(exc))
            return

        self._set_trade_controls_enabled(True)
        self.log_output.clear()
        self._append_log(f"초기 현금 {initial_cash:,.0f}원으로 가상 거래를 시작합니다.")
        self._update_balance_label()
        self.positions_table.setRowCount(0)

    def _set_trade_controls_enabled(self, enabled: bool) -> None:
        self.symbol_input.setEnabled(enabled)
        self.price_input.setEnabled(enabled)
        self.quantity_input.setEnabled(enabled)
        self.buy_button.setEnabled(enabled)
        self.sell_button.setEnabled(enabled)
        self.hold_button.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Mock trading helpers
    # ------------------------------------------------------------------
    def _ensure_order_client(self) -> KiwoomOrderClient:
        if self.order_client is None:
            self.order_client = KiwoomOrderClient()
        return self.order_client

    def _authenticate_mock(self) -> None:
        appkey = self.appkey_input.text().strip()
        secret = self.secret_input.text().strip()
        if not appkey or not secret:
            QMessageBox.warning(self, "입력 필요", "모의 App Key와 Secret Key를 모두 입력해주세요.")
            return

        client = self._ensure_order_client()
        client.set_credentials(appkey, secret)
        try:
            client.authenticate()
        except Exception as exc:  # pragma: no cover - network
            QMessageBox.critical(self, "모의투자 인증 실패", str(exc))
            self._append_log(f"[Mock API] 인증 실패: {exc}")
            return

        QMessageBox.information(self, "모의투자 인증", "모의투자 액세스 토큰이 발급되었습니다.")
        self._append_log("[Mock API] 인증 완료")

    def _build_order_payload(self, symbol: str, price: float, quantity: int) -> Dict[str, Any]:
        account = self.account_input.text().strip()
        if not account:
            raise ValueError("계좌번호를 입력해주세요.")
        product_code = self.product_code_input.text().strip() or "01"
        order_division = self.order_division_input.text().strip() or "00"
        return {
            "CANO": account,
            "ACNT_PRDT_CD": product_code,
            "PDNO": symbol,
            "ORD_DVSN": order_division,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": f"{price:.2f}",
        }

    def _fetch_account_snapshot(self) -> None:
        client = self._ensure_order_client()
        if not client.access_token:
            QMessageBox.warning(self, "모의투자 인증", "먼저 모의투자 인증을 진행해주세요.")
            return
        endpoint = self.balance_endpoint_input.text().strip() or "/api/dostk/acnt"
        tr_id = self.balance_tr_id_input.text().strip()
        if not tr_id:
            QMessageBox.warning(self, "입력 필요", "계좌 조회용 TR ID를 입력해주세요.")
            return
        api_id = self.balance_api_id_input.text().strip() or None
        custtype = self.custtype_input.text().strip() or "P"
        try:
            payload = self._build_account_payload()
        except ValueError as exc:
            QMessageBox.warning(self, "입력 필요", str(exc))
            return
        try:
            response = client.fetch_account_summary(
                endpoint=endpoint,
                tr_id=tr_id,
                custtype=custtype,
                payload=payload,
                api_id=api_id,
            )
        except Exception as exc:
            QMessageBox.critical(self, "계좌 조회 실패", str(exc))
            self._append_log(f"[Account] 조회 실패: {exc}")
            return
        pretty = json.dumps(response, ensure_ascii=False, indent=2)
        self._append_log(f"[Account]\n{pretty}")
        summary_text = self._format_account_summary(response)
        self.account_snapshot_label.setText(summary_text)

    def _build_account_payload(self) -> Dict[str, Any]:
        account = self.account_input.text().strip()
        if not account:
            raise ValueError("계좌번호를 입력해주세요.")
        product_code = self.product_code_input.text().strip() or "01"
        return {
            "CANO": account,
            "ACNT_PRDT_CD": product_code,
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
        summary = None
        for key in ("output", "output1", "result"):
            value = payload.get(key)
            if isinstance(value, dict):
                summary = value
                break
        if summary is None:
            summary = payload
        d2 = _safe_decimal(summary.get("d2_deposit") or summary.get("d2_entra_amt"))
        eval_amt = _safe_decimal(summary.get("tot_evalu_amt") or summary.get("tot_evlu_amt"))
        cash = _safe_decimal(summary.get("prsm_dpst") or summary.get("prsm_dpst_aset_amt"))
        total_pnl = _safe_decimal(summary.get("tot_pnl_amt") or summary.get("lspft_amt"))
        return f"계좌 정보: 예수금 {d2:,.0f}원 / 평가금 {eval_amt:,.0f}원 / 현금 {cash:,.0f}원 / 손익 {total_pnl:,.0f}원"

    def _send_order_to_api(self, side: str, symbol: str, price: float, quantity: int) -> None:
        endpoint = self.order_endpoint_input.text().strip()
        tr_input = self.buy_tr_id_input if side == "BUY" else self.sell_tr_id_input
        tr_id = tr_input.text().strip()
        if not endpoint or not tr_id:
            self._append_log(f"[Mock API] {side} 요청 건너뜀 (endpoint / TR ID 미입력)")
            return

        client = self._ensure_order_client()
        if not client.access_token:
            self._append_log("[Mock API] 액세스 토큰이 없어 주문을 전송하지 않았습니다.")
            return

        try:
            payload = self._build_order_payload(symbol, price, quantity)
        except ValueError as exc:
            self._append_log(f"[Mock API] {side} 주문 실패: {exc}")
            return

        custtype = self.custtype_input.text().strip() or "P"
        api_id = self.order_api_id_input.text().strip() or None
        try:
            response = client.place_order(
                endpoint=endpoint,
                tr_id=tr_id,
                custtype=custtype,
                payload=payload,
                api_id=api_id,
            )
        except Exception as exc:  # pragma: no cover - network
            self._append_log(f"[Mock API] {side} 주문 실패: {exc}")
            return

        summary = response.get("msg1") or response.get("message") or json.dumps(response, ensure_ascii=False)
        self._append_log(f"[Mock API] {side} 주문 성공: {summary}")

    def _trigger_auto_training(self) -> None:
        if not AI_MODULE_AVAILABLE:
            QMessageBox.warning(self, "AI 모듈 없음", "ai 패키지가 설치되지 않아 자동 학습을 실행할 수 없습니다.")
            return
        if self.auto_trainer and self.auto_trainer.isRunning():
            QMessageBox.information(self, "학습 진행 중", "이미 학습이 진행 중입니다.")
            return
        self.auto_train_button.setEnabled(False)
        self.ai_status_label.setText("AI 상태: 학습 중 ...")
        self._append_log("[AI] Auto Learn을 시작합니다.")
        self.auto_trainer = AutoTrainWorker(parent=self, ticks=60, top_k=10, timesteps=50_000)
        self.auto_trainer.progress.connect(self._append_log)
        self.auto_trainer.completed.connect(self._auto_training_completed)
        self.auto_trainer.failed.connect(self._auto_training_failed)
        self.auto_trainer.start()

    def _auto_training_completed(self, tickers: List[str]) -> None:
        self.auto_train_button.setEnabled(True)
        names = ", ".join(tickers) if tickers else "없음"
        self._append_log(f"[AI] 학습 완료. 선택 종목: {names}")
        self.ai_status_label.setText(f"AI 상태: 완료 (K={len(tickers)})")
        self.auto_trainer = None

    def _auto_training_failed(self, message: str) -> None:
        self.auto_train_button.setEnabled(True)
        self._append_log(f"[AI] 학습 실패: {message}")
        self.ai_status_label.setText("AI 상태: 실패")
        self.auto_trainer = None

    # ------------------------------------------------------------------
    # Trade handlers
    # ------------------------------------------------------------------
    # Trade handlers
    # ------------------------------------------------------------------
    def _require_broker(self) -> VirtualBroker:
        if self.broker is None:
            raise RuntimeError("가상 거래를 시작해주세요.")
        return self.broker

    def _buy(self) -> None:
        self._execute_trade("BUY")

    def _sell(self) -> None:
        self._execute_trade("SELL")

    def _hold(self) -> None:
        broker = self._require_broker()
        symbol = self.symbol_input.text().strip()
        broker.hold(symbol)
        self._log_last_history()
        self._update_balance_label()

    def _execute_trade(self, side: str) -> None:
        broker = self._require_broker()
        try:
            symbol, price, quantity = self._parse_trade_inputs()
            if side == "BUY":
                broker.buy(symbol, price, quantity)
            else:
                broker.sell(symbol, price, quantity)
        except Exception as exc:  # pragma: no cover - PyQt runtime errors
            title = "�ż� ����" if side == "BUY" else "�ŵ� ����"
            QMessageBox.warning(self, title, str(exc))
            return
        self._log_last_history()
        self._update_positions_table()
        self._update_balance_label()
        self._send_order_to_api(side, symbol, price, quantity)

    def _parse_trade_inputs(self) -> tuple[str, float, int]:
        symbol = self.symbol_input.text().strip()
        if not symbol:
            raise ValueError("종목 코드를 입력해주세요.")
        try:
            price = float(self.price_input.text())
        except ValueError:
            raise ValueError("가격은 숫자여야 합니다.")
        try:
            quantity = int(self.quantity_input.text())
        except ValueError:
            raise ValueError("수량은 정수여야 합니다.")
        return symbol, price, quantity

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _append_log(self, message: str) -> None:
        self.log_output.append(message)

    def _log_last_history(self) -> None:
        broker = self._require_broker()
        if broker.history:
            self._append_log(broker.history[-1])

    def _update_balance_label(self) -> None:
        broker = self._require_broker()
        self.balance_label.setText(
            "현금: {cash:,.0f}원 / 평가금액: {value:,.0f}원 / 총손익: {pnl:,.0f}원".format(
                cash=broker.cash,
                value=broker.market_value,
                pnl=broker.total_pnl,
            )
        )

    def _update_positions_table(self) -> None:
        broker = self._require_broker()
        positions = sorted(broker.positions.values(), key=lambda pos: pos.symbol)
        self.positions_table.setRowCount(len(positions))
        for row, position in enumerate(positions):
            self.positions_table.setItem(row, 0, QTableWidgetItem(position.symbol))
            self.positions_table.setItem(row, 1, self._format_number_item(position.quantity))
            self.positions_table.setItem(row, 2, self._format_number_item(position.avg_price))
            self.positions_table.setItem(row, 3, self._format_number_item(position.last_price))
            self.positions_table.setItem(row, 4, self._format_number_item(position.market_value))
            self.positions_table.setItem(row, 5, self._format_number_item(position.unrealised_pnl))

    @staticmethod
    def _format_number_item(value: Union[float, int]) -> QTableWidgetItem:
        if isinstance(value, int):
            text = f"{value:,}"
        else:
            text = f"{value:,.2f}"
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return item



class AutoTrainWorker(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, *, ticks: int, top_k: int, timesteps: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ticks = ticks
        self.top_k = top_k
        self.timesteps = timesteps
        self.fees = {"buy": 0.00015, "sell": 0.00015, "tax_sell": 0.0005}
        self.slippage = 5.0

    def run(self) -> None:
        if not AI_MODULE_AVAILABLE:
            self.failed.emit("AI 모듈이 설치되어 있지 않습니다.")
            return
        try:
            self.progress.emit("[AI] Synthetic 데이터 생성 중 ...")
            data = generate_synthetic_universe(num_tickers=max(self.top_k * 2, 20), num_minutes=5_000)
            wf_cfg = WalkForwardConfig(horizon_ticks=self.ticks)
            selector_cfg = SelectorConfig(top_k=self.top_k, horizon_ticks=self.ticks)
            bundle = prepare_datasets(data, wf_cfg)
            self.progress.emit("[AI] Stage-A Selector 학습 중 ...")
            result = run_selector_stage(bundle, selector_cfg, wf_cfg)
            arrays, prices = self._build_arrays(data, bundle, result.selected)
            if not arrays:
                raise RuntimeError("선택된 종목이 없습니다.")
            self.progress.emit(f"[AI] Stage-B RL 환경 구성 (K={len(arrays)}) ...")
            envs = make_vec_envs(arrays, prices, window=self.ticks, fees=self.fees, slippage_bps=self.slippage)
            rl_cfg = MockTrainingConfig(total_timesteps=self.timesteps, window=self.ticks)
            train_rl_agent(envs, rl_cfg)
            self.completed.emit(list(result.selected))
        except Exception as exc:
            self.failed.emit(str(exc))

    def _build_arrays(self, data, bundle, selected):
        arrays = []
        prices = []
        for ticker in selected:
            try:
                feat = bundle.features.xs(ticker)
            except KeyError:
                continue
            feat_values = feat.values.astype(np.float32)
            raw = data.xs(ticker)
            aligned = raw.loc[feat.index, "close"].values.astype(np.float32)
            if len(feat_values) > self.ticks:
                arrays.append(feat_values)
                prices.append(aligned)
        return arrays, prices


def run() -> None:
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    window = VirtualTradingWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
