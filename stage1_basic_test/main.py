from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
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


TRACKED_HEADERS = ["next-key", "cont-yn", "api-id"]


def _parse_float(value: Any) -> float:
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


def _parse_int(value: Any) -> int:
    return int(round(_parse_float(value)))


@dataclass
class OverseasSummary:
    krw_estimated_asset: float
    evaluation_amount: float
    purchase_amount: float


@dataclass
class OverseasHolding:
    symbol: str
    name: str
    quantity: float
    buy_amount: float
    eval_amount: float
    average_price: float
    current_price: float


class KiwoomRestClient:
    """Minimal Kiwoom REST client for account balance TR calls."""

    def __init__(self, *, mode: str = "real") -> None:
        if mode not in {"real", "mock"}:
            raise ValueError("mode must be 'real' or 'mock'")
        self.mode = mode
        self.host = "https://api.kiwoom.com" if mode == "real" else "https://mockapi.kiwoom.com"
        self._access_token: Optional[str] = None
        self._appkey: Optional[str] = None
        self._appsecret: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_credentials(self, appkey: str, secretkey: str) -> None:
        self._appkey = appkey.strip()
        self._appsecret = secretkey.strip()

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

    def fetch_overseas_balance(
        self,
        payload: Dict[str, Any],
        *,
        endpoint: str,
        api_id: str,
        tr_id: Optional[str] = None,
        custtype: str = "P",
        cont_yn: str = "N",
        next_key: str = "",
    ) -> Tuple[OverseasSummary, List[OverseasHolding], Dict[str, Any]]:
        if self.mode == "mock":
            summary, holdings = self._mock_overseas_balance()
            return summary, holdings, {
                "mode": "mock",
                "summary": summary.__dict__,
                "holdings": [h.__dict__ for h in holdings],
            }

        if not self._access_token:
            raise RuntimeError("No access token. Call authenticate() first.")

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
            raise RuntimeError("Overseas balance request failed.")

        summary, holdings = self._parse_overseas_balance(response)
        return summary, holdings, response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
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

    def _parse_overseas_balance(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[OverseasSummary, List[OverseasHolding]]:
        summary_source = self._find_first_mapping(
            payload,
            ["output", "output1", "summary", "data", "result"],
        )

        summary = OverseasSummary(
            krw_estimated_asset=_parse_float(self._find_first(summary_source, [
                "krw_estimated_asset",
                "ovrs_kor_estm_amt",
                "ovrs_tot_estm_amt",
                "ovrs_kor_evlt_amt",
            ])),
            evaluation_amount=_parse_float(self._find_first(summary_source, [
                "evaluation_amount",
                "ovrs_evlt_amt",
                "ovrs_pdls_amt",
            ])),
            purchase_amount=_parse_float(self._find_first(summary_source, [
                "purchase_amount",
                "ovrs_buy_amt",
                "ovrs_pdls_buy_amt",
            ])),
        )

        holdings_source = self._find_first_list(
            payload,
            ["output1", "output2", "stocks", "holdings", "items", "result_list"],
        )

        holdings: List[OverseasHolding] = []
        for entry in holdings_source:
            holdings.append(
                OverseasHolding(
                    symbol=str(self._find_first(entry, ["symbol", "ovrs_item_cd", "stk_cd", "code"]) or "-").strip(),
                    name=str(self._find_first(entry, ["name", "ovrs_item_nm", "item_name", "stock_name"]) or "-").strip(),
                    quantity=_parse_float(self._find_first(entry, ["quantity", "ovrs_cblc_qty", "qty", "hold_qty"])),
                    buy_amount=_parse_float(self._find_first(entry, ["buy_amount", "ovrs_buamt", "buy_amt", "pchs_amt"])),
                    eval_amount=_parse_float(self._find_first(entry, ["eval_amount", "ovrs_evlt_amt", "eval_amt", "evlt_amt"])),
                    average_price=_parse_float(self._find_first(entry, ["average_price", "ovrs_avg_prc", "avg_price", "avg_prc"])),
                    current_price=_parse_float(self._find_first(entry, ["current_price", "ovrs_now_prc", "prpr", "current_prc"])),
                )
            )

        return summary, holdings

    @staticmethod
    def _find_first_mapping(payload: Dict[str, Any], candidates: List[str]) -> Dict[str, Any]:
        for key in candidates:
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        return {}

    @staticmethod
    def _find_first(source: Dict[str, Any] | None, candidates: List[str]) -> Any:
        if not source:
            return None
        for key in candidates:
            if key in source:
                return source[key]
        return None

    @staticmethod
    def _find_first_list(payload: Dict[str, Any], candidates: List[str]) -> List[Dict[str, Any]]:
        for key in candidates:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    def _mock_overseas_balance(self) -> Tuple[OverseasSummary, List[OverseasHolding]]:
        summary = OverseasSummary(
            krw_estimated_asset=52_350_000.0,
            evaluation_amount=49_800_000.0,
            purchase_amount=45_900_000.0,
        )
        holdings = [
            OverseasHolding(
                symbol="AAPL",
                name="Apple",
                quantity=30,
                buy_amount=4_500_000.0,
                eval_amount=5_100_000.0,
                average_price=150_000.0,
                current_price=170_000.0,
            ),
            OverseasHolding(
                symbol="TSLA",
                name="Tesla",
                quantity=10,
                buy_amount=7_800_000.0,
                eval_amount=6_900_000.0,
                average_price=780_000.0,
                current_price=690_000.0,
            ),
        ]
        return summary, holdings


class Stage1Window(QMainWindow):
    """GUI for querying overseas balance and holdings through the Kiwoom REST API."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StockAI - Stage 1 Overseas Balance Viewer")
        self.resize(860, 620)

        self.current_appkey: str = ""
        self.current_secret: str = ""

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["real", "mock"])
        self.mode_combo.currentTextChanged.connect(self._mode_changed)

        self.appkey_input = QLineEdit()
        self.secret_input = QLineEdit()
        self.secret_input.setEchoMode(QLineEdit.Password)

        self.endpoint_input = QLineEdit("/api/overseas/balance")
        self.api_id_input = QLineEdit("kt00004")
        self.api_id_input.setPlaceholderText("api-id header value, e.g. kt00004")
        self.tr_header_input = QLineEdit()
        self.tr_header_input.setPlaceholderText("Optional tr_id header, e.g. TTTS3004R")
        self.custtype_input = QLineEdit("P")
        self.custtype_input.setMaxLength(1)
        self.custtype_input.setPlaceholderText("Customer type (P or B)")

        self.payload_input = QTextEdit()
        self.payload_input.setPlainText(
            json.dumps(
                {
                    "CANO": "00000000",
                    "ACNT_PRDT_CD": "01",
                    "OVRS_EXCG_CD": "NASD",
                    "TR_CCY_CD": "USD",
                },
                indent=4,
            )
        )

        self.authenticate_button = QPushButton("Get Access Token")
        self.authenticate_button.clicked.connect(self._authenticate)

        self.fetch_button = QPushButton("Fetch Overseas Balance")
        self.fetch_button.clicked.connect(self._fetch_balance)

        self.summary_labels = {
            "krw_estimated_asset": QLabel("-"),
            "evaluation_amount": QLabel("-"),
            "purchase_amount": QLabel("-"),
        }

        summary_grid = QGridLayout()
        summary_grid.addWidget(QLabel("KRW Estimated Asset"), 0, 0)
        summary_grid.addWidget(self.summary_labels["krw_estimated_asset"], 0, 1)
        summary_grid.addWidget(QLabel("Evaluation Amount"), 1, 0)
        summary_grid.addWidget(self.summary_labels["evaluation_amount"], 1, 1)
        summary_grid.addWidget(QLabel("Purchase Amount"), 2, 0)
        summary_grid.addWidget(self.summary_labels["purchase_amount"], 2, 1)
        summary_box = QGroupBox("Summary")
        summary_box.setLayout(summary_grid)

        self.holdings_table = QTableWidget(0, 7)
        self.holdings_table.setHorizontalHeaderLabels(
            [
                "Symbol",
                "Name",
                "Quantity",
                "Buy Amount",
                "Evaluation Amount",
                "Average Price",
                "Current Price",
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

        self.client = KiwoomRestClient(mode=self.mode_combo.currentText())

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _mode_changed(self, mode: str) -> None:
        self.client = KiwoomRestClient(mode=mode)
        if self.current_appkey or self.current_secret:
            self.client.set_credentials(self.current_appkey, self.current_secret)
        if mode == "mock":
            self.raw_output.setPlainText("Running in mock mode. Real network calls are disabled.")
        else:
            self.raw_output.clear()

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
            QMessageBox.warning(self, "Missing TR ID", "Enter the TR ID (e.g. kt00004).")
            return

        try:
            summary, holdings, raw = self.client.fetch_overseas_balance(
                payload,
                endpoint=endpoint,
                api_id=api_id,
                tr_id=tr_id_header,
                custtype=custtype,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Request Failed", str(exc))
            self.raw_output.setPlainText(str(exc))
            return

        self._update_summary(summary)
        self._update_holdings_table(holdings)
        self.raw_output.setPlainText(json.dumps(raw, indent=4, ensure_ascii=False))

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _update_summary(self, summary: OverseasSummary) -> None:
        self.summary_labels["krw_estimated_asset"].setText(f"{summary.krw_estimated_asset:,.0f}")
        self.summary_labels["evaluation_amount"].setText(f"{summary.evaluation_amount:,.0f}")
        self.summary_labels["purchase_amount"].setText(f"{summary.purchase_amount:,.0f}")

    def _update_holdings_table(self, holdings: List[OverseasHolding]) -> None:
        self.holdings_table.setRowCount(len(holdings))
        for row, item in enumerate(holdings):
            data = [
                item.symbol,
                item.name,
                f"{item.quantity:,.2f}",
                f"{item.buy_amount:,.0f}",
                f"{item.eval_amount:,.0f}",
                f"{item.average_price:,.2f}",
                f"{item.current_price:,.2f}",
            ]
            for column, value in enumerate(data):
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
