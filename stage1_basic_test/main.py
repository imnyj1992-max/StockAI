from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
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


def _parse_decimal(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if not stripped:
            return 0.0
        try:
            return float(stripped)
        except ValueError:
            return 0.0
    return 0.0


def _format_quantity(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _format_currency(value: float) -> str:
    return f"{value:,.0f}"


def _format_price(value: float) -> str:
    return f"{value:,.0f}"


def _format_rate(value: float) -> str:
    return f"{value:.2f}%"


@dataclass
class AccountSummary:
    account_name: str
    branch_name: str
    d2_deposit: float
    total_evaluation: float
    total_purchase: float
    deposit_assets: float
    total_profit_loss: float
    total_profit_loss_rate: float
    daily_profit_loss: float
    daily_profit_loss_rate: float


@dataclass
class AccountHolding:
    symbol: str
    name: str
    quantity: float
    purchase_amount: float
    evaluation_amount: float
    average_price: float
    current_price: float
    profit_loss: float
    profit_loss_rate: float


# Optional hashkey behavior is controlled at runtime; hashkey is not always required.


class KiwoomRestClient:
    """REST client to communicate with Kiwoom domestic account APIs (e.g. kt00004)."""

    def __init__(self, *, mode: str = "real", require_hashkey: bool = False) -> None:
        if mode not in {"real", "mock"}:
            raise ValueError("mode must be 'real' or 'mock'")
        self.mode = mode
        self.host = "https://api.kiwoom.com" if mode == "real" else "https://mockapi.kiwoom.com"
        self._access_token: Optional[str] = None
        self._appkey: Optional[str] = None
        self._appsecret: Optional[str] = None
        self._hashkey_cache: Dict[str, str] = {}
        self.last_hashkey_error: Optional[str] = None
        self.require_hashkey = require_hashkey

    def set_credentials(self, appkey: str, secretkey: str) -> None:
        self._appkey = appkey.strip()
        self._appsecret = secretkey.strip()
        self._hashkey_cache.clear()
        self.last_hashkey_error = None

    def set_hashkey_required(self, required: bool) -> None:
        self.require_hashkey = required

    def authenticate(self, appkey: str, secretkey: str) -> str:
        self.set_credentials(appkey, secretkey)
        payload = {
            "grant_type": "client_credentials",
            "appkey": appkey,
            "secretkey": secretkey,
        }
        response = self._post("/oauth2/token", payload)
        if response is None:
            raise RuntimeError("Token request failed.")

        token = response.get("access_token") or response.get("token") or response.get("accessToken")
        if not token:
            raise RuntimeError("Access token missing in response.")
        self._access_token = token
        return token

    def fetch_account_balance(
        self,
        payload: Dict[str, Any],
        *,
        endpoint: str,
        api_id: str,
        tr_id: Optional[str] = None,
        custtype: str = "P",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> Tuple[AccountSummary, List[AccountHolding], Dict[str, Any]]:
        self.last_hashkey_error = None
        if self.mode == "mock":
            summary, holdings, raw = self._mock_account_balance()
            return summary, holdings, raw

        if not self._access_token:
            raise RuntimeError("No access token. Authenticate first.")

        response = self._post(
            endpoint,
            payload,
            token=self._access_token,
            api_id=api_id,
            tr_id=tr_id,
            custtype=custtype,
            cont_yn=cont_yn,
            next_key=next_key,
        )
        if response is None:
            raise RuntimeError("Account balance request failed.")

        summary, holdings = self._parse_account_balance(response)
        return summary, holdings, response

    def _post(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        *,
        token: Optional[str] = None,
        api_id: Optional[str] = None,
        tr_id: Optional[str] = None,
        custtype: Optional[str] = None,
        cont_yn: str = "N",
        next_key: str = "",
    ) -> Optional[Dict[str, Any]]:
        url = self.host + endpoint
        headers = {"Content-Type": "application/json;charset=UTF-8"}

        payload_json = json.dumps(payload, ensure_ascii=False)

        if token:
            headers["authorization"] = f"Bearer {token}"
            headers["cont-yn"] = cont_yn
            headers["next-key"] = next_key
        if api_id:
            headers["api-id"] = api_id
        if tr_id:
            headers["tr_id"] = tr_id
        if custtype:
            headers["custtype"] = custtype
        if self._appkey:
            headers["appkey"] = self._appkey
        if self._appsecret:
            headers["appsecret"] = self._appsecret

        if token and payload and self.mode == "real" and self.require_hashkey:
            hashkey = self._generate_hashkey(payload_json)
            if hashkey:
                headers["hashkey"] = hashkey

        data = payload_json.encode("utf-8")

        try:
            response = requests.post(url, headers=headers, data=data, timeout=10)
        except requests.RequestException as exc:
            raise RuntimeError(f"HTTP request failed: {exc}") from exc

        try:
            body = response.json()
        except ValueError:
            body = None

        if response.status_code >= 400:
            message = response.text
            if isinstance(body, dict):
                message = (
                    body.get("message")
                    or body.get("msg")
                    or json.dumps(body, ensure_ascii=False)
                )
            raise RuntimeError(f"HTTP {response.status_code}: {message}")

        if not isinstance(body, dict):
            raise RuntimeError("Response is not a JSON object.")

        return body

    def _generate_hashkey(self, payload_json: str) -> Optional[str]:
        if not self._appkey or not self._appsecret:
            return None
        cache_key = payload_json
        self.last_hashkey_error = None
        if cache_key in self._hashkey_cache:
            return self._hashkey_cache[cache_key]

        url = self.host + "/oauth2/hashkey"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "appkey": self._appkey,
            "appsecret": self._appsecret,
        }
        if self._access_token:
            headers["authorization"] = f"Bearer {self._access_token}"
        try:
            response = requests.post(url, headers=headers, data=payload_json.encode("utf-8"), timeout=10)
        except requests.RequestException as exc:
            raise RuntimeError(f"Hashkey request failed: {exc}") from exc

        try:
            body = response.json()
        except ValueError:
            body = None

        if response.status_code >= 400:
            message = None
            if isinstance(body, dict):
                message = body.get("message") or body.get("msg")
            if not message:
                message = response.text
            if response.status_code >= 500:
                # Hashkey may be optional for read-only TRs; keep a note and continue without it.
                self.last_hashkey_error = message or "Unexpected server error during hashkey generation."
                return None
            raise RuntimeError(f"Hashkey request failed ({response.status_code}): {message}")

        if not isinstance(body, dict):
            raise RuntimeError("Hashkey response is not a JSON object.")

        key = body.get("hashkey") or body.get("HASHKEY")
        if not key:
            raise RuntimeError("Hashkey not found in response.")

        self._hashkey_cache[cache_key] = key
        return key

    def _parse_account_balance(self, payload: Dict[str, Any]) -> Tuple[AccountSummary, List[AccountHolding]]:
        summary = AccountSummary(
            account_name=str(payload.get("acnt_nm", "-") or "-"),
            branch_name=str(payload.get("brch_nm", "-") or "-"),
            d2_deposit=_parse_decimal(payload.get("d2_entra")),
            total_evaluation=_parse_decimal(payload.get("aset_evlt_amt")),
            total_purchase=_parse_decimal(payload.get("tot_pur_amt")),
            deposit_assets=_parse_decimal(payload.get("prsm_dpst_aset_amt")),
            total_profit_loss=_parse_decimal(payload.get("lspft_amt")),
            total_profit_loss_rate=_parse_decimal(payload.get("lspft_rt")),
            daily_profit_loss=_parse_decimal(payload.get("tdy_lspft_amt")),
            daily_profit_loss_rate=_parse_decimal(payload.get("tdy_lspft_rt")),
        )

        holdings: List[AccountHolding] = []
        rows = payload.get("stk_acnt_evlt_prst")
        if isinstance(rows, list):
            for entry in rows:
                if not isinstance(entry, dict):
                    continue
                symbol = str(entry.get("stk_cd", "")).strip()
                if symbol.startswith("A") and len(symbol) > 1:
                    symbol = symbol[1:]
                holdings.append(
                    AccountHolding(
                        symbol=symbol or "-",
                        name=str(entry.get("stk_nm", "-") or "-"),
                        quantity=_parse_decimal(entry.get("rmnd_qty")),
                        purchase_amount=_parse_decimal(entry.get("pur_amt")),
                        evaluation_amount=_parse_decimal(entry.get("evlt_amt")),
                        average_price=_parse_decimal(entry.get("avg_prc")),
                        current_price=_parse_decimal(entry.get("cur_prc")),
                        profit_loss=_parse_decimal(entry.get("pl_amt")),
                        profit_loss_rate=_parse_decimal(entry.get("pl_rt")),
                    )
                )

        return summary, holdings

    def _mock_account_balance(self) -> Tuple[AccountSummary, List[AccountHolding], Dict[str, Any]]:
        sample_response = {
            "acnt_nm": "김키움",
            "brch_nm": "키움은행",
            "d2_entra": "000000012550",
            "aset_evlt_amt": "000000761950",
            "tot_pur_amt": "000000002786",
            "prsm_dpst_aset_amt": "000000749792",
            "lspft_amt": "000000000000",
            "lspft_rt": "0.00",
            "tdy_lspft_amt": "000000000000",
            "tdy_lspft_rt": "0.00",
            "stk_acnt_evlt_prst": [
                {
                    "stk_cd": "A005930",
                    "stk_nm": "삼성전자",
                    "rmnd_qty": "000000000003",
                    "avg_prc": "000000124500",
                    "cur_prc": "000000070000",
                    "evlt_amt": "000000209542",
                    "pl_amt": "-00000163958",
                    "pl_rt": "-43.8977",
                    "pur_amt": "000000373500",
                }
            ],
            "return_code": 0,
            "return_msg": "조회가 완료되었습니다.",
        }
        summary, holdings = self._parse_account_balance(sample_response)
        return summary, holdings, sample_response


class Stage1Window(QMainWindow):
    """PyQt5 GUI to display domestic account holdings via kt00004."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StockAI - Stage 1: Account Evaluation (kt00004)")
        self.resize(900, 640)

        self.current_appkey: str = ""
        self.current_secret: str = ""

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["real", "mock"])
        self.mode_combo.currentTextChanged.connect(self._mode_changed)

        self.appkey_input = QLineEdit()
        self.secret_input = QLineEdit()
        self.secret_input.setEchoMode(QLineEdit.Password)

        self.endpoint_input = QLineEdit("/api/dostk/acnt")
        self.api_id_input = QLineEdit("kt00004")
        self.api_id_input.setPlaceholderText("api-id header value")
        self.tr_header_input = QLineEdit("TTTS3004R")
        self.tr_header_input.setPlaceholderText("Optional tr_id header")
        self.custtype_input = QLineEdit("P")
        self.custtype_input.setMaxLength(1)
        self.custtype_input.setPlaceholderText("Customer type (P or B)")

        self.payload_input = QTextEdit()
        self.payload_input.setPlainText(
            json.dumps(
                {
                    "qry_tp": "0",
                    "dmst_stex_tp": "KRX",
                },
                indent=4,
                ensure_ascii=False,
            )
        )

        self.authenticate_button = QPushButton("Get Access Token")
        self.authenticate_button.clicked.connect(self._authenticate)

        self.fetch_button = QPushButton("Fetch Account Balance")
        self.fetch_button.clicked.connect(self._fetch_balance)

        self.hashkey_checkbox = QCheckBox("Include hashkey header (if required)")
        self.hashkey_checkbox.setChecked(False)
        self.hashkey_checkbox.stateChanged.connect(self._toggle_hashkey)

        self.summary_labels: Dict[str, QLabel] = {
            "account_name": QLabel("-"),
            "branch_name": QLabel("-"),
            "d2_deposit": QLabel("-"),
            "total_evaluation": QLabel("-"),
            "total_purchase": QLabel("-"),
            "deposit_assets": QLabel("-"),
            "total_profit_loss": QLabel("-"),
            "total_profit_loss_rate": QLabel("-"),
            "daily_profit_loss": QLabel("-"),
            "daily_profit_loss_rate": QLabel("-"),
        }

        summary_grid = QGridLayout()
        summary_grid.addWidget(QLabel("Account Name"), 0, 0)
        summary_grid.addWidget(self.summary_labels["account_name"], 0, 1)
        summary_grid.addWidget(QLabel("Branch"), 0, 2)
        summary_grid.addWidget(self.summary_labels["branch_name"], 0, 3)

        summary_grid.addWidget(QLabel("D+2 Deposit"), 1, 0)
        summary_grid.addWidget(self.summary_labels["d2_deposit"], 1, 1)
        summary_grid.addWidget(QLabel("Deposit Assets"), 1, 2)
        summary_grid.addWidget(self.summary_labels["deposit_assets"], 1, 3)

        summary_grid.addWidget(QLabel("Total Evaluation"), 2, 0)
        summary_grid.addWidget(self.summary_labels["total_evaluation"], 2, 1)
        summary_grid.addWidget(QLabel("Total Purchase"), 2, 2)
        summary_grid.addWidget(self.summary_labels["total_purchase"], 2, 3)

        summary_grid.addWidget(QLabel("Total P/L"), 3, 0)
        summary_grid.addWidget(self.summary_labels["total_profit_loss"], 3, 1)
        summary_grid.addWidget(QLabel("Total P/L %"), 3, 2)
        summary_grid.addWidget(self.summary_labels["total_profit_loss_rate"], 3, 3)

        summary_grid.addWidget(QLabel("Today's P/L"), 4, 0)
        summary_grid.addWidget(self.summary_labels["daily_profit_loss"], 4, 1)
        summary_grid.addWidget(QLabel("Today's P/L %"), 4, 2)
        summary_grid.addWidget(self.summary_labels["daily_profit_loss_rate"], 4, 3)

        summary_box = QGroupBox("Account Summary")
        summary_box.setLayout(summary_grid)

        self.holdings_table = QTableWidget(0, 9)
        self.holdings_table.setHorizontalHeaderLabels(
            [
                "Symbol",
                "Name",
                "Quantity",
                "Purchase Amount",
                "Evaluation Amount",
                "Average Price",
                "Current Price",
                "Profit/Loss",
                "Profit/Loss %",
            ]
        )
        self.holdings_table.horizontalHeader().setStretchLastSection(True)
        self.holdings_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.raw_output = QTextEdit()
        self.raw_output.setReadOnly(True)
        self.raw_output.setPlaceholderText("Raw API response will appear here.")

        form_layout = QFormLayout()
        form_layout.addRow("Mode", self.mode_combo)
        form_layout.addRow("App Key", self.appkey_input)
        form_layout.addRow("Secret Key", self.secret_input)
        form_layout.addRow("Endpoint", self.endpoint_input)
        form_layout.addRow("API ID", self.api_id_input)
        form_layout.addRow("TR Header (tr_id)", self.tr_header_input)
        form_layout.addRow("Customer Type", self.custtype_input)
        form_widget = QWidget()
        form_widget.setLayout(form_layout)

        button_row = QGridLayout()
        button_row.addWidget(self.authenticate_button, 0, 0)
        button_row.addWidget(self.fetch_button, 0, 1)
        button_row.addWidget(self.hashkey_checkbox, 1, 0, 1, 2)

        payload_box = QGroupBox("Request Payload (JSON)")
        payload_layout = QVBoxLayout()
        payload_layout.addWidget(self.payload_input)
        payload_box.setLayout(payload_layout)

        control_box = QGroupBox("Configuration")
        control_layout = QVBoxLayout()
        control_layout.addWidget(form_widget)
        control_layout.addLayout(button_row)
        control_layout.addWidget(payload_box)
        control_box.setLayout(control_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(control_box)
        main_layout.addWidget(summary_box)
        main_layout.addWidget(self.holdings_table)
        main_layout.addWidget(QLabel("Raw Response"))
        main_layout.addWidget(self.raw_output)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.client = KiwoomRestClient(
            mode=self.mode_combo.currentText(), require_hashkey=self.hashkey_checkbox.isChecked()
        )

    def _mode_changed(self, mode: str) -> None:
        self.client = KiwoomRestClient(mode=mode, require_hashkey=self.hashkey_checkbox.isChecked())
        if self.current_appkey or self.current_secret:
            self.client.set_credentials(self.current_appkey, self.current_secret)
        if mode == "mock":
            self.raw_output.setPlainText("Running in mock mode. Real network calls are disabled.")
        else:
            self.raw_output.clear()

    def _toggle_hashkey(self) -> None:
        self.client.set_hashkey_required(self.hashkey_checkbox.isChecked())

    def _authenticate(self) -> None:
        appkey = self.appkey_input.text().strip()
        secret = self.secret_input.text().strip()
        if not appkey or not secret:
            QMessageBox.warning(self, "Missing Credentials", "Enter both app key and secret key.")
            return

        self.current_appkey = appkey
        self.current_secret = secret
        self.client.set_credentials(appkey, secret)

        if self.client.mode == "mock":
            QMessageBox.information(self, "Mock Mode", "Mock mode does not require authentication.")
            return

        try:
            token = self.client.authenticate(appkey, secret)
        except Exception as exc:
            QMessageBox.critical(self, "Authentication Failed", str(exc))
            self.raw_output.setPlainText(str(exc))
            return

        QMessageBox.information(self, "Authentication", "Access token acquired successfully.")
        self.raw_output.setPlainText(f"Access token acquired: {token[:6]}... (hidden)")

    def _fetch_balance(self) -> None:
        try:
            payload = json.loads(self.payload_input.toPlainText())
            if not isinstance(payload, dict):
                raise ValueError("Payload root must be a JSON object.")
        except json.JSONDecodeError as exc:
            QMessageBox.critical(self, "Invalid Payload", f"JSON decode error: {exc}")
            return
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid Payload", str(exc))
            return

        endpoint = self.endpoint_input.text().strip()
        api_id = self.api_id_input.text().strip()
        tr_id_header = self.tr_header_input.text().strip() or None
        custtype = self.custtype_input.text().strip().upper() or "P"
        if len(custtype) > 1:
            custtype = custtype[0]

        if not endpoint.startswith("/"):
            QMessageBox.warning(self, "Invalid Endpoint", "Endpoint should start with '/' (e.g. /api/...).")
            return
        if not api_id:
            QMessageBox.warning(self, "Missing API ID", "Enter the api-id value (e.g. kt00004).")
            return

        try:
            summary, holdings, raw = self.client.fetch_account_balance(
                payload,
                endpoint=endpoint,
                api_id=api_id,
                tr_id=tr_id_header,
                custtype=custtype,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Request Failed", str(exc))
            warning = self.client.last_hashkey_error
            text = str(exc)
            if warning:
                text += f"\n\n[Hashkey warning] {warning}"
            self.raw_output.setPlainText(text)
            return

        self._update_summary(summary)
        self._update_holdings_table(holdings)
        output_text = json.dumps(raw, indent=4, ensure_ascii=False)
        warning = self.client.last_hashkey_error
        if warning:
            output_text += f"\n\n[Hashkey warning] {warning}"
        self.raw_output.setPlainText(output_text)

    def _update_summary(self, summary: AccountSummary) -> None:
        self.summary_labels["account_name"].setText(summary.account_name or "-")
        self.summary_labels["branch_name"].setText(summary.branch_name or "-")
        self.summary_labels["d2_deposit"].setText(_format_currency(summary.d2_deposit))
        self.summary_labels["deposit_assets"].setText(_format_currency(summary.deposit_assets))
        self.summary_labels["total_evaluation"].setText(_format_currency(summary.total_evaluation))
        self.summary_labels["total_purchase"].setText(_format_currency(summary.total_purchase))
        self.summary_labels["total_profit_loss"].setText(_format_currency(summary.total_profit_loss))
        self.summary_labels["total_profit_loss_rate"].setText(_format_rate(summary.total_profit_loss_rate))
        self.summary_labels["daily_profit_loss"].setText(_format_currency(summary.daily_profit_loss))
        self.summary_labels["daily_profit_loss_rate"].setText(_format_rate(summary.daily_profit_loss_rate))

    def _update_holdings_table(self, holdings: List[AccountHolding]) -> None:
        self.holdings_table.setRowCount(len(holdings))
        for row, item in enumerate(holdings):
            values = [
                item.symbol,
                item.name,
                _format_quantity(item.quantity),
                _format_currency(item.purchase_amount),
                _format_currency(item.evaluation_amount),
                _format_price(item.average_price),
                _format_price(item.current_price),
                _format_currency(item.profit_loss),
                _format_rate(item.profit_loss_rate),
            ]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column >= 2:
                    cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.holdings_table.setItem(row, column, cell)


def run() -> None:
    app = QApplication(sys.argv)
    window = Stage1Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
