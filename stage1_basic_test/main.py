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
    purchase_amount: float
    evaluation_amount: float
    average_price: float
    current_price: float


class KiwoomRestClient:
    """Minimal Kiwoom REST client supporting domestic and overseas balance TRs."""

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
        self.last_hashkey_error = None
        if self.mode == "mock":
            summary, holdings, raw = self._mock_overseas_balance()
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
            raise RuntimeError("Overseas balance request failed.")

        summary, holdings = self._parse_overseas_balance(response)
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

    def _parse_overseas_balance(self, payload: Dict[str, Any]) -> Tuple[OverseasSummary, List[OverseasHolding]]:
        summary_source = self._find_first_mapping(
            payload,
            ["output", "output1", "summary", "data", "result"],
        )

        summary = OverseasSummary(
            krw_estimated_asset=_parse_decimal(
                self._find_first(
                    summary_source,
                    [
                        "krw_estimated_asset",
                        "ovrs_kor_estm_amt",
                        "ovrs_tot_estm_amt",
                        "kor_tot_evlt_amt",
                        "ovrs_kor_evlt_amt",
                    ],
                )
            ),
            evaluation_amount=_parse_decimal(
                self._find_first(
                    summary_source,
                    ["evaluation_amount", "ovrs_evlt_amt", "ovrs_pdls_amt", "evlt_amt"],
                )
            ),
            purchase_amount=_parse_decimal(
                self._find_first(
                    summary_source,
                    ["purchase_amount", "ovrs_buy_amt", "ovrs_pdls_buy_amt", "buy_amt"],
                )
            ),
        )

        holdings_source = self._find_first_list(
            payload,
            ["output1", "output2", "stocks", "holdings", "items", "result_list"],
        )

        holdings: List[OverseasHolding] = []
        for entry in holdings_source:
            symbol = str(self._find_first(entry, ["stk_cd", "ovrs_item_cd", "code", "symbol"]) or "").strip()
            if symbol.startswith("A") and len(symbol) > 1:
                symbol = symbol[1:]
            holdings.append(
                OverseasHolding(
                    symbol=symbol or "-",
                    name=str(
                        self._find_first(entry, ["stk_nm", "ovrs_item_nm", "name", "item_name"]) or "-"
                    ).strip()
                    or "-",
                    quantity=_parse_decimal(
                        self._find_first(entry, ["rmnd_qty", "ovrs_cblc_qty", "qty", "hold_qty"])
                    ),
                    purchase_amount=_parse_decimal(
                        self._find_first(entry, ["pur_amt", "ovrs_buamt", "buy_amt", "pchs_amt"])
                    ),
                    evaluation_amount=_parse_decimal(
                        self._find_first(entry, ["evlt_amt", "ovrs_evlt_amt", "eval_amt", "evlt_amt"])
                    ),
                    average_price=_parse_decimal(
                        self._find_first(entry, ["avg_prc", "ovrs_avg_prc", "avg_price", "avg_prc"])
                    ),
                    current_price=_parse_decimal(
                        self._find_first(entry, ["cur_prc", "ovrs_now_prc", "prpr", "current_price"])
                    ),
                )
            )

        return summary, holdings

    def _mock_account_balance(self) -> Tuple[AccountSummary, List[AccountHolding], Dict[str, Any]]:
        sample_response = {
            "acnt_nm": "Sample User",
            "brch_nm": "Sample Branch",
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
                    "stk_nm": "SAMSUNG ELEC",
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

    def _mock_overseas_balance(self) -> Tuple[OverseasSummary, List[OverseasHolding], Dict[str, Any]]:
        sample_response = {
            "krw_estimated_asset": "00000052350000",
            "ovrs_evlt_amt": "00000049800000",
            "ovrs_buy_amt": "00000045900000",
            "holdings": [
                {
                    "stk_cd": "AAPL",
                    "stk_nm": "Apple",
                    "rmnd_qty": "30",
                    "avg_prc": "150000",
                    "cur_prc": "170000",
                    "evlt_amt": "5100000",
                    "pur_amt": "4500000",
                },
                {
                    "stk_cd": "TSLA",
                    "stk_nm": "Tesla",
                    "rmnd_qty": "10",
                    "avg_prc": "780000",
                    "cur_prc": "690000",
                    "evlt_amt": "6900000",
                    "pur_amt": "7800000",
                },
            ],
        }
        summary, holdings = self._parse_overseas_balance(sample_response)
        return summary, holdings, sample_response

    @staticmethod
    def _find_first_mapping(payload: Dict[str, Any], candidates: List[str]) -> Dict[str, Any]:
        for key in candidates:
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        return payload

    @staticmethod
    def _find_first(source: Optional[Dict[str, Any]], candidates: List[str]) -> Any:
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


class Stage1Window(QMainWindow):
    DATA_TYPES = ("Domestic Stocks", "Overseas Stocks")
    DOMESTIC_COLUMNS = [
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
    OVERSEAS_COLUMNS = [
        "Symbol",
        "Name",
        "Quantity",
        "Purchase Amount",
        "Evaluation Amount",
        "Average Price",
        "Current Price",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StockAI - Stage 1: Account Overview")
        self.resize(960, 660)

        self.current_appkey: str = ""
        self.current_secret: str = ""
        self.client: Optional[KiwoomRestClient] = None

        domestic_payload = json.dumps(
            {
                "qry_tp": "0",
                "dmst_stex_tp": "KRX",
            },
            indent=4,
            ensure_ascii=False,
        )
        overseas_payload = json.dumps(
            {
                "CANO": "00000000",
                "ACNT_PRDT_CD": "01",
                "OVRS_EXCG_CD": "NASD",
                "TR_CCY_CD": "USD",
            },
            indent=4,
            ensure_ascii=False,
        )

        self.saved_configs: Dict[str, Dict[str, Any]] = {
            "Domestic Stocks": {
                "endpoint": "/api/dostk/acnt",
                "api_id": "kt00004",
                "tr_id": "TTTS3004R",
                "custtype": "P",
                "payload": domestic_payload,
                "hashkey": False,
            },
            "Overseas Stocks": {
                "endpoint": "/api/overseas/balance",
                "api_id": "",
                "tr_id": "",
                "custtype": "P",
                "payload": overseas_payload,
                "hashkey": False,
            },
        }

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["real", "mock"])
        self.mode_combo.currentTextChanged.connect(self._mode_changed)

        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(self.DATA_TYPES)
        self.data_type_combo.currentTextChanged.connect(self._data_type_changed)

        self.appkey_input = QLineEdit("eNfziSc7ibOOl8tqrI8-K-sjl9o3oKI6QpzaZANWopM")
        self.secret_input = QLineEdit("Ei1bzCY-vO2cY4WMTty8-fJx0JmvDFJuYS7q4PUDQgs")
        self.secret_input.setEchoMode(QLineEdit.Password)

        self.endpoint_input = QLineEdit()
        self.api_id_input = QLineEdit()
        self.api_id_input.setPlaceholderText("api-id header value")
        self.tr_header_input = QLineEdit()
        self.tr_header_input.setPlaceholderText("Optional tr_id header")
        self.custtype_input = QLineEdit()
        self.custtype_input.setMaxLength(1)
        self.custtype_input.setPlaceholderText("Customer type (P or B)")

        self.payload_input = QTextEdit()

        self.authenticate_button = QPushButton("Get Access Token")
        self.authenticate_button.clicked.connect(self._authenticate)

        self.fetch_button = QPushButton("Fetch Account Balance")
        self.fetch_button.clicked.connect(self._fetch_balance)

        self.hashkey_checkbox = QCheckBox("Include hashkey header (if required)")
        self.hashkey_checkbox.stateChanged.connect(self._toggle_hashkey)

        self.summary_layout = QGridLayout()
        summary_box = QGroupBox("Account Summary")
        summary_box.setLayout(self.summary_layout)
        self._set_summary_rows([])

        self.holdings_table = QTableWidget()
        self.holdings_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.holdings_table.horizontalHeader().setStretchLastSection(True)
        self.current_columns: List[str] = []
        self._configure_holdings_table(self.DOMESTIC_COLUMNS)

        self.raw_output = QTextEdit()
        self.raw_output.setReadOnly(True)
        self.raw_output.setPlaceholderText("Raw API response will appear here.")

        form_layout = QFormLayout()
        form_layout.addRow("Mode", self.mode_combo)
        form_layout.addRow("Data Type", self.data_type_combo)
        form_layout.addRow("App Key", self.appkey_input)
        form_layout.addRow("Secret Key", self.secret_input)
        form_layout.addRow("Endpoint", self.endpoint_input)
        form_layout.addRow("API ID", self.api_id_input)
        form_layout.addRow("TR Header (tr_id)", self.tr_header_input)
        form_layout.addRow("Customer Type", self.custtype_input)
        form_widget = QWidget()
        form_widget.setLayout(form_layout)

        button_layout = QGridLayout()
        button_layout.addWidget(self.authenticate_button, 0, 0)
        button_layout.addWidget(self.fetch_button, 0, 1)
        button_layout.addWidget(self.hashkey_checkbox, 1, 0, 1, 2)

        payload_box = QGroupBox("Request Payload (JSON)")
        payload_layout = QVBoxLayout()
        payload_layout.addWidget(self.payload_input)
        payload_box.setLayout(payload_layout)

        control_box = QGroupBox("Configuration")
        control_layout = QVBoxLayout()
        control_layout.addWidget(form_widget)
        control_layout.addLayout(button_layout)
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

        self.current_data_type = self.DATA_TYPES[0]
        self._load_config(self.current_data_type)
        self._create_client()

    def _create_client(self) -> None:
        require_hashkey = self.hashkey_checkbox.isChecked() and self._is_domestic_selected()
        self.client = KiwoomRestClient(mode=self.mode_combo.currentText(), require_hashkey=require_hashkey)
        if self.current_appkey or self.current_secret:
            self.client.set_credentials(self.current_appkey, self.current_secret)

    def _is_domestic_selected(self) -> bool:
        return self.current_data_type == "Domestic Stocks"

    def _save_current_config(self, data_type: str) -> None:
        self.saved_configs[data_type] = {
            "endpoint": self.endpoint_input.text().strip(),
            "api_id": self.api_id_input.text().strip(),
            "tr_id": self.tr_header_input.text().strip(),
            "custtype": self.custtype_input.text().strip() or "P",
            "payload": self.payload_input.toPlainText(),
            "hashkey": self.hashkey_checkbox.isChecked() if data_type == "Domestic Stocks" else False,
        }

    def _load_config(self, data_type: str) -> None:
        config = self.saved_configs[data_type]
        self.endpoint_input.setText(config.get("endpoint", ""))
        self.api_id_input.setText(config.get("api_id", ""))
        self.tr_header_input.setText(config.get("tr_id", ""))
        self.custtype_input.setText(config.get("custtype", "P"))
        self.payload_input.setPlainText(config.get("payload", "{}"))

        is_domestic = data_type == "Domestic Stocks"
        self.hashkey_checkbox.blockSignals(True)
        self.hashkey_checkbox.setEnabled(is_domestic)
        self.hashkey_checkbox.setChecked(config.get("hashkey", False) if is_domestic else False)
        self.hashkey_checkbox.blockSignals(False)
        if self.client:
            self.client.set_hashkey_required(self.hashkey_checkbox.isChecked() and is_domestic)

        columns = self.DOMESTIC_COLUMNS if is_domestic else self.OVERSEAS_COLUMNS
        self._configure_holdings_table(columns)
        self._set_summary_rows([])

    def _configure_holdings_table(self, columns: List[str]) -> None:
        self.current_columns = columns
        self.holdings_table.clear()
        self.holdings_table.setColumnCount(len(columns))
        self.holdings_table.setHorizontalHeaderLabels(columns)
        self.holdings_table.setRowCount(0)
        self.holdings_table.horizontalHeader().setStretchLastSection(True)

    def _set_summary_rows(self, rows: List[Tuple[str, str]]) -> None:
        while self.summary_layout.count():
            item = self.summary_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, (label_text, value_text) in enumerate(rows):
            row = index // 2
            column = (index % 2) * 2
            self.summary_layout.addWidget(QLabel(label_text), row, column)
            self.summary_layout.addWidget(QLabel(value_text), row, column + 1)

    def _mode_changed(self, mode: str) -> None:
        self._create_client()
        if mode == "mock":
            self.raw_output.setPlainText("Running in mock mode. Real network calls are disabled.")
        else:
            self.raw_output.clear()

    def _data_type_changed(self, new_type: str) -> None:
        self._save_current_config(self.current_data_type)
        self.current_data_type = new_type
        self._load_config(new_type)
        if self.client:
            self.client.set_hashkey_required(self.hashkey_checkbox.isChecked() and self._is_domestic_selected())
        if self.mode_combo.currentText() == "mock":
            self.raw_output.setPlainText("Running in mock mode. Real network calls are disabled.")
        else:
            self.raw_output.clear()

    def _toggle_hashkey(self) -> None:
        checked = self.hashkey_checkbox.isChecked()
        self.saved_configs[self.current_data_type]["hashkey"] = checked if self._is_domestic_selected() else False
        if self.client:
            self.client.set_hashkey_required(checked and self._is_domestic_selected())

    def _authenticate(self) -> None:
        appkey = self.appkey_input.text().strip()
        secret = self.secret_input.text().strip()
        if not appkey or not secret:
            QMessageBox.warning(self, "Missing Credentials", "Enter both app key and secret key.")
            return

        self.current_appkey = appkey
        self.current_secret = secret
        if not self.client:
            self._create_client()
        if self.client:
            self.client.set_credentials(appkey, secret)

        if self.mode_combo.currentText() == "mock":
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
        self._save_current_config(self.current_data_type)

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
        tr_id = self.tr_header_input.text().strip() or None
        custtype = self.custtype_input.text().strip() or "P"

        if not endpoint.startswith("/"):
            QMessageBox.warning(self, "Invalid Endpoint", "Endpoint should start with '/' (e.g. /api/...).")
            return
        if not api_id:
            QMessageBox.warning(self, "Missing API ID", "Enter the api-id value for the TR you are calling.")
            return

        if not self.client:
            self._create_client()

        try:
            if self._is_domestic_selected():
                summary, holdings, raw = self.client.fetch_account_balance(
                    payload,
                    endpoint=endpoint,
                    api_id=api_id,
                    tr_id=tr_id,
                    custtype=custtype,
                )
                self._update_domestic_summary(summary)
                self._update_domestic_holdings(holdings)
            else:
                summary, holdings, raw = self.client.fetch_overseas_balance(
                    payload,
                    endpoint=endpoint,
                    api_id=api_id,
                    tr_id=tr_id,
                    custtype=custtype,
                )
                self._update_overseas_summary(summary)
                self._update_overseas_holdings(holdings)
        except Exception as exc:
            QMessageBox.critical(self, "Request Failed", str(exc))
            warning = self.client.last_hashkey_error if self.client else None
            message = str(exc)
            if warning:
                message += f"\n\n[Hashkey warning] {warning}"
            self.raw_output.setPlainText(message)
            return

        output_text = json.dumps(raw, indent=4, ensure_ascii=False)
        warning = self.client.last_hashkey_error if self.client else None
        if warning:
            output_text += f"\n\n[Hashkey warning] {warning}"
        self.raw_output.setPlainText(output_text)

    def _update_domestic_summary(self, summary: AccountSummary) -> None:
        rows = [
            ("Account Name", summary.account_name or "-"),
            ("Branch", summary.branch_name or "-"),
            ("D+2 Deposit", _format_currency(summary.d2_deposit)),
            ("Deposit Assets", _format_currency(summary.deposit_assets)),
            ("Total Evaluation", _format_currency(summary.total_evaluation)),
            ("Total Purchase", _format_currency(summary.total_purchase)),
            ("Total P/L", _format_currency(summary.total_profit_loss)),
            ("Total P/L %", _format_rate(summary.total_profit_loss_rate)),
            ("Today's P/L", _format_currency(summary.daily_profit_loss)),
            ("Today's P/L %", _format_rate(summary.daily_profit_loss_rate)),
        ]
        self._set_summary_rows(rows)

    def _update_overseas_summary(self, summary: OverseasSummary) -> None:
        rows = [
            ("KRW Estimated Asset", _format_currency(summary.krw_estimated_asset)),
            ("Evaluation Amount", _format_currency(summary.evaluation_amount)),
            ("Purchase Amount", _format_currency(summary.purchase_amount)),
        ]
        self._set_summary_rows(rows)

    def _update_domestic_holdings(self, holdings: List[AccountHolding]) -> None:
        self._configure_holdings_table(self.DOMESTIC_COLUMNS)
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

    def _update_overseas_holdings(self, holdings: List[OverseasHolding]) -> None:
        self._configure_holdings_table(self.OVERSEAS_COLUMNS)
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
