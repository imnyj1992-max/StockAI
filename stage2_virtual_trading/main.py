"""Stage 2: Virtual trading simulator GUI.

This module builds on top of the project scaffolding from stage 1 but
replaces the Kiwoom OpenAPI dependency with a fully self-contained virtual
broker.  The simulator allows the user to initialise a cash balance and then
perform buy/sell/hold actions to validate the GUI flow before connecting to
real trading infrastructure.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
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


class VirtualTradingWindow(QMainWindow):
    """Main GUI window for stage 2 virtual trading."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StockAI - Stage 2: Virtual Trading Simulator")
        self.resize(720, 520)

        self.broker: Optional[VirtualBroker] = None

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

        self.balance_label = QLabel("현금: - / 평가금액: - / 손익: -")

        self.positions_table = QTableWidget(0, 6)
        self.positions_table.setHorizontalHeaderLabels(
            ["종목", "수량", "평균 단가", "현재가", "평가금액", "평가손익"]
        )
        self.positions_table.horizontalHeader().setStretchLastSection(True)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        # Layout setup
        control_layout = QGridLayout()
        control_layout.addWidget(QLabel("초기 현금"), 0, 0)
        control_layout.addWidget(self.initial_cash_input, 0, 1)
        control_layout.addWidget(self.start_button, 0, 2)

        trade_layout = QHBoxLayout()
        trade_layout.addWidget(self.symbol_input)
        trade_layout.addWidget(self.price_input)
        trade_layout.addWidget(self.quantity_input)
        trade_layout.addWidget(self.buy_button)
        trade_layout.addWidget(self.sell_button)
        trade_layout.addWidget(self.hold_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addLayout(trade_layout)
        main_layout.addWidget(self.balance_label)
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
        self.log_output.append(f"초기 현금 {initial_cash:,.0f}원으로 가상 거래를 시작합니다.")
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
    # Trade handlers
    # ------------------------------------------------------------------
    def _require_broker(self) -> VirtualBroker:
        if self.broker is None:
            raise RuntimeError("가상 거래를 시작해주세요.")
        return self.broker

    def _buy(self) -> None:
        broker = self._require_broker()
        try:
            symbol, price, quantity = self._parse_trade_inputs()
            broker.buy(symbol, price, quantity)
        except Exception as exc:  # pragma: no cover - PyQt runtime errors
            QMessageBox.warning(self, "매수 오류", str(exc))
            return
        self._log_last_history()
        self._update_positions_table()
        self._update_balance_label()

    def _sell(self) -> None:
        broker = self._require_broker()
        try:
            symbol, price, quantity = self._parse_trade_inputs()
            broker.sell(symbol, price, quantity)
        except Exception as exc:  # pragma: no cover
            QMessageBox.warning(self, "매도 오류", str(exc))
            return
        self._log_last_history()
        self._update_positions_table()
        self._update_balance_label()

    def _hold(self) -> None:
        broker = self._require_broker()
        symbol = self.symbol_input.text().strip()
        broker.hold(symbol)
        self._log_last_history()
        self._update_balance_label()

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
    def _log_last_history(self) -> None:
        broker = self._require_broker()
        if broker.history:
            self.log_output.append(broker.history[-1])

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


def run() -> None:
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    window = VirtualTradingWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
