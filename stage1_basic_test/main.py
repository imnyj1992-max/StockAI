"""Stage 1: Basic Kiwoom OpenAPI login and account information GUI.

This module defines a PyQt5 GUI that performs the following tasks:
- Initiates a login request through the Kiwoom OpenAPI+ control.
- Displays the login status to the user.
- Retrieves the first connected account number and basic account balance information.
- Presents the fetched account data in a simple GUI.

The code is written to run on Windows where Kiwoom OpenAPI+ is available. On
other platforms the ActiveX control will not be accessible.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Optional

from PyQt5.QtCore import QCoreApplication, QEventLoop, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QAxContainer import QAxWidget


@dataclass
class AccountBalance:
    """Simple container for account balance summary information."""

    account_number: str
    deposit: str
    available_funds: str
    estimated_value: str

    @classmethod
    def from_raw(cls, account: str, data: Dict[str, str]) -> "AccountBalance":
        return cls(
            account_number=account,
            deposit=data.get("예수금", ""),
            available_funds=data.get("출금가능금액", ""),
            estimated_value=data.get("추정예탁자산", ""),
        )


class KiwoomOpenApiWidget(QAxWidget):
    """Wrapper around the Kiwoom OpenAPI ActiveX control."""

    def __init__(self) -> None:
        super().__init__()

        if not self.setControl("KHOPENAPI.KHOpenAPICtrl.1"):
            raise RuntimeError(
                "Kiwoom OpenAPI+ 컨트롤을 불러오지 못했습니다. "
                "키움 OpenAPI+가 설치되어 있고 32비트 Python 환경에서 실행 중인지 확인해주세요."
            )

        self.login_event_loop: Optional[QEventLoop] = None
        self.comm_event_loop: Optional[QEventLoop] = None

        try:
            self.OnEventConnect.connect(self._on_login)
            self.OnReceiveTrData.connect(self._on_receive_tr_data)
        except AttributeError as exc:
            raise RuntimeError(
                "Kiwoom OpenAPI+ 이벤트를 초기화하지 못했습니다. "
                "키움 OpenAPI+가 올바르게 설치되었는지 확인해주세요."
            ) from exc

        self.accounts: list[str] = []
        self.account_info: Dict[str, Dict[str, str]] = {}
        self.api_code: str = ""


    # ------------------------------------------------------------------
    # Login handling
    # ------------------------------------------------------------------
    def login(self) -> None:
        self.dynamicCall("CommConnect()")
        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def set_personal_api_code(self, api_code: str) -> None:
        """Store the user-provided API code for later authenticated requests."""

        self.api_code = api_code.strip()

        if not self.api_code:
            return

        # Some environments expose helper methods for registering the API key.
        # Because the availability differs by installation, any failure is
        # silenced so the GUI continues operating with the stored value.
        try:  # pragma: no cover - depends on native Kiwoom installation
            self.dynamicCall(
                "KOA_Functions(QString, QString)", "SetApiKey", self.api_code
            )
        except Exception:
            pass
    def _on_login(self, err_code: int) -> None:
        if self.login_event_loop is None:
            return
        self.login_event_loop.exit()
        self.login_event_loop = None

    # ------------------------------------------------------------------
    # Account retrieval
    # ------------------------------------------------------------------
    def fetch_accounts(self) -> list[str]:
        accounts = self.dynamicCall("GetLoginInfo(QString)", "ACCNO")
        if not accounts:
            return []
        self.accounts = [acc for acc in accounts.split(";") if acc]
        return self.accounts

    def request_account_balance(self, account: str) -> None:
        self.SetInputValue("계좌번호", account)
        password = self.api_code or "0000"
        self.SetInputValue("비밀번호", password)
        self.SetInputValue("비밀번호입력매체구분", "00")
        self.SetInputValue("조회구분", "2")
        self.comm_event_loop = QEventLoop()
        self.CommRqData("opw00001_req", "opw00001", 0, "1000")
        self.comm_event_loop.exec_()

    def _on_receive_tr_data(
        self,
        screen_no: str,
        rqname: str,
        trcode: str,
        recordname: str,
        prev_next: str,
        data_len: int,
        err_code: str,
        msg1: str,
        msg2: str,
    ) -> None:
        if rqname == "opw00001_req":
            account = self.accounts[0] if self.accounts else ""
            self.account_info[account] = {
                "예수금": self._comm_get_data(rqname, trcode, 0, "예수금"),
                "출금가능금액": self._comm_get_data(rqname, trcode, 0, "출금가능금액"),
                "추정예탁자산": self._comm_get_data(rqname, trcode, 0, "추정예탁자산"),
            }
        if self.comm_event_loop is not None:
            self.comm_event_loop.exit()
            self.comm_event_loop = None

    def _comm_get_data(self, rqname: str, trcode: str, idx: int, item_name: str) -> str:
        return self.dynamicCall(
            "CommGetData(QString, QString, QString, int, QString)",
            trcode,
            "",
            rqname,
            idx,
            item_name,
        ).strip()


class MainWindow(QMainWindow):
    """Main GUI window for stage 1 test."""

    def __init__(self, kiwoom: KiwoomOpenApiWidget) -> None:
        super().__init__()
        self.kiwoom = kiwoom

        self.setWindowTitle("StockAI - Stage 1: Account Viewer")
        self.resize(480, 360)

        self.status_label = QLabel("로그인 상태: 미로그인")
        self.api_code_label = QLabel("개인 API 코드 (계좌 비밀번호)")
        self.api_code_input = QLineEdit()
        self.api_code_input.setPlaceholderText("API 코드 또는 계좌 비밀번호")
        self.api_code_input.setEchoMode(QLineEdit.Password)
        self.account_input = QLineEdit()
        self.account_input.setPlaceholderText("계좌번호")
        self.account_input.setReadOnly(True)

        self.login_button = QPushButton("로그인")
        self.login_button.clicked.connect(self._login)

        self.fetch_button = QPushButton("계좌 조회")
        self.fetch_button.clicked.connect(self._fetch_account)
        self.fetch_button.setEnabled(False)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.api_code_label)
        layout.addWidget(self.api_code_input)
        layout.addWidget(self.account_input)
        layout.addWidget(self.login_button)
        layout.addWidget(self.fetch_button)
        layout.addWidget(self.output)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _login(self) -> None:
        self.kiwoom.set_personal_api_code(self.api_code_input.text())
        try:
            self.kiwoom.login()
        except Exception as exc:  # pragma: no cover - ActiveX errors on unsupported OS
            QMessageBox.critical(self, "로그인 실패", str(exc))
            return

        accounts = self.kiwoom.fetch_accounts()
        if not accounts:
            QMessageBox.warning(self, "계좌 없음", "등록된 계좌를 찾을 수 없습니다.")
            return

        first_account = accounts[0]
        self.account_input.setText(first_account)
        self.status_label.setText("로그인 상태: 로그인 완료")
        self.fetch_button.setEnabled(True)
        self.output.append("로그인에 성공했습니다. 계좌를 선택해주세요.")

    def _fetch_account(self) -> None:
        account = self.account_input.text().strip()
        if not account:
            QMessageBox.warning(self, "계좌 선택", "조회할 계좌를 선택해주세요.")
            return

        try:
            self.kiwoom.request_account_balance(account)
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "조회 실패", str(exc))
            return

        data = self.kiwoom.account_info.get(account)
        if not data:
            QMessageBox.warning(self, "데이터 없음", "계좌 정보를 가져오지 못했습니다.")
            return

        balance = AccountBalance.from_raw(account, data)
        self._display_account_info(balance)

    def _display_account_info(self, balance: AccountBalance) -> None:
        self.output.clear()
        self.output.append(f"계좌번호: {balance.account_number}")
        self.output.append(f"예수금: {balance.deposit}")
        self.output.append(f"출금가능금액: {balance.available_funds}")
        self.output.append(f"추정예탁자산: {balance.estimated_value}")


def run() -> None:
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)

    try:
        kiwoom = KiwoomOpenApiWidget()
    except RuntimeError as exc:
        QMessageBox.critical(None, "Kiwoom OpenAPI 오류", str(exc))
        sys.exit(1)
    window = MainWindow(kiwoom)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
