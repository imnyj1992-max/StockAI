# Stage 1 — Basic Kiwoom OpenAPI Test

이 단계에서는 다음을 목표로 합니다.

1. 키움증권 OpenAPI+ 로그인 테스트
2. 연결된 계좌 번호 확인
3. 계좌 잔액 및 기초 정보를 조회하여 GUI에 표시

## 실행 방법

> ⚠️ **중요:** 키움증권 OpenAPI+는 Windows 환경에서만 ActiveX 컨트롤을 통해 동작합니다. 따라서 Windows + Python 3.9~3.11(32비트) 환경에서 실행해야 하며, 키움 OpenAPI가 설치되어 있어야 합니다. (64비트 Python에서는 ActiveX 컨트롤이 로드되지 않습니다.)

1. PyQt5 및 필요 라이브러리 설치

```bash
pip install pyqt5
```

2. `stage1_basic_test/main.py` 실행

```bash
python stage1_basic_test/main.py
```

3. 로그인 전에 개인 API 코드(또는 계좌 비밀번호)를 입력한 뒤 로그인 버튼을 눌러 키움증권 계정에 로그인합니다.
4. 계좌 조회 버튼으로 선택된 계좌의 잔액 정보를 확인합니다.

## 주요 구성 요소

- `KiwoomOpenApiWidget`: ActiveX 컨트롤을 래핑하여 로그인 및 TR 데이터를 요청하는 클래스
- `MainWindow`: PyQt5 기반 GUI, 로그인 및 계좌 조회 버튼과 결과 표시 창을 제공하며 API 코드 입력 필드를 포함합니다.

## 향후 단계 연계

- 2단계에서는 가상 거래를 위한 시뮬레이터 UI를 추가하고, 계좌/잔액 부분은 여기서 구축한 컴포넌트를 재활용하여 확장할 수 있습니다.
