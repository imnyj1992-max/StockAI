import requests
import json

mode = 'real'  # real: 실전투자, mock: 모의투자

host = 'https://api.kiwoom.com' # 실전투자
if mode == 'mock': host = 'https://mockapi.kiwoom.com' # 모의투자

def 접근토큰_발급(data):
	url =  host + '/oauth2/token'
	headers = {'Content-Type': 'application/json;charset=UTF-8',}
	response = requests.post(url, headers=headers, json=data)
	print('Code:', response.status_code)
	print('Header:', json.dumps({key: response.headers.get(key) for key in ['next-key', 'cont-yn', 'api-id']}, indent=4, ensure_ascii=False))
	print('Body:', json.dumps(response.json(), indent=4, ensure_ascii=False))
	return response

def 공매도추이요청(token, data, cont_yn='N', next_key=''):
	url =  host + '/api/dostk/shsa'
	# 컨텐츠타입, 접근토큰, 연속조회여부, 연속조회키, TR명
	headers = {'Content-Type': 'application/json;charset=UTF-8', 'authorization': f'Bearer {token}', 'cont-yn': cont_yn, 'next-key': next_key, 'api-id': 'ka10014',}
	response = requests.post(url, headers=headers, json=data)
	print('Code:', response.status_code)
	print('Header:', json.dumps({key: response.headers.get(key) for key in ['next-key', 'cont-yn', 'api-id']}, indent=4, ensure_ascii=False))
	print('Body:', json.dumps(response.json(), indent=4, ensure_ascii=False))

def 주식외국인종목별매매동향(token, data, cont_yn='N', next_key=''):
	url =  host + '/api/dostk/frgnistt'
	headers = {'Content-Type': 'application/json;charset=UTF-8', 'authorization': f'Bearer {token}', 'cont-yn': cont_yn, 'next-key': next_key, 'api-id': 'ka10008',}
	response = requests.post(url, headers=headers, json=data)
	print('Code:', response.status_code)
	print('Header:', json.dumps({key: response.headers.get(key) for key in ['next-key', 'cont-yn', 'api-id']}, indent=4, ensure_ascii=False))
	print('Body:', json.dumps(response.json(), indent=4, ensure_ascii=False))

def 대차거래추이요청(token, data, cont_yn='N', next_key=''):
	url =  host + '/api/dostk/slb'
	headers = {'Content-Type': 'application/json;charset=UTF-8', 'authorization': f'Bearer {token}', 'cont-yn': cont_yn, 'next-key': next_key, 'api-id': 'ka10068',}
	response = requests.post(url, headers=headers, json=data)
	print('Code:', response.status_code)
	print('Header:', json.dumps({key: response.headers.get(key) for key in ['next-key', 'cont-yn', 'api-id']}, indent=4, ensure_ascii=False))
	print('Body:', json.dumps(response.json(), indent=4, ensure_ascii=False))
	
    
def 호가잔량상위요청(token, data, cont_yn='N', next_key=''):
	url =  host + '/api/dostk/rkinfo'
	headers = {'Content-Type': 'application/json;charset=UTF-8', 'authorization': f'Bearer {token}', 'cont-yn': cont_yn, 'next-key': next_key, 'api-id': 'ka10020',}
	response = requests.post(url, headers=headers, json=data)
	print('Code:', response.status_code)
	print('Header:', json.dumps({key: response.headers.get(key) for key in ['next-key', 'cont-yn', 'api-id']}, indent=4, ensure_ascii=False))
	print('Body:', json.dumps(response.json(), indent=4, ensure_ascii=False))


if __name__ == '__main__':
	params = {
		'grant_type': 'client_credentials',  # grant_type
		'appkey': 'eNfziSc7ibOOl8tqrI8-K-sjl9o3oKI6QpzaZANWopM',  # 앱키
		'secretkey': 'Ei1bzCY-vO2cY4WMTty8-fJx0JmvDFJuYS7q4PUDQgs',  # 시크릿키
	}
	access_token = 접근토큰_발급(data=params)
	MY_ACCESS_TOKEN = access_token.json().get('token')  # 접근토큰
	    
	params = {
		'stk_cd': '005930', # 종목코드 거래소별 종목코드 (KRX:039490,NXT:039490_NX,SOR:039490_AL)
		'tm_tp': '1', # 시간구분 0:시작일, 1:기간
		'strt_dt': '20250501', # 시작일자 YYYYMMDD
		'end_dt': '20250519', # 종료일자 YYYYMMDD
	}
	공매도추이요청(token=MY_ACCESS_TOKEN, data=params)
	# next-key, cont-yn 값이 있을 경우
	# 공매도추이요청(token=MY_ACCESS_TOKEN, data=params, cont_yn='Y', next_key='nextkey..')

	params = {'stk_cd': '005930', }# 종목코드 거래소별 종목코드 (KRX:039490,NXT:039490_NX,SOR:039490_AL)
	주식외국인종목별매매동향(token=MY_ACCESS_TOKEN, data=params)
	# next-key, cont-yn 값이 있을 경우
	# 주식외국인종목별매매동향(token=MY_ACCESS_TOKEN, data=params, cont_yn='Y', next_key='nextkey..')
	
	params = {
		'strt_dt': '20250401', # 시작일자 YYYYMMDD
		'end_dt': '20250430', # 종료일자 YYYYMMDD
		'all_tp': '1', # 전체구분 1: 전체표시
	}
	대차거래추이요청(token=MY_ACCESS_TOKEN, data=params)
	# next-key, cont-yn 값이 있을 경우
	# 대차거래추이요청(token=MY_ACCESS_TOKEN, data=params, cont_yn='Y', next_key='nextkey..')
	
	params = {
		'mrkt_tp': '001', # 시장구분 001:코스피, 101:코스닥
		'sort_tp': '1', # 정렬구분 1:순매수잔량순, 2:순매도잔량순, 3:매수비율순, 4:매도비율순
		'trde_qty_tp': '0000', # 거래량구분 0000:장시작전(0주이상), 0010:만주이상, 0050:5만주이상, 00100:10만주이상
		'stk_cnd': '0', # 종목조건 0:전체조회, 1:관리종목제외, 5:증100제외, 6:증100만보기, 7:증40만보기, 8:증30만보기, 9:증20만보기
		'crd_cnd': '0', # 신용조건 0:전체조회, 1:신용융자A군, 2:신용융자B군, 3:신용융자C군, 4:신용융자D군, 7:신용융자E군, 9:신용융자전체
		'stex_tp': '1', # 거래소구분 1:KRX, 2:NXT 3.통합
	}
	호가잔량상위요청(token=MY_ACCESS_TOKEN, data=params)
	# next-key, cont-yn 값이 있을 경우
	# 호가잔량상위요청(token=MY_ACCESS_TOKEN, data=params, cont_yn='Y', next_key='nextkey..')
	
    