import requests
try:
    r = requests.post('http://127.0.0.1:8082/assign',
                      json={'customer_id': 'test-2', 'ratio': 0.5}, timeout=5)
    print('status', r.status_code)
    print('text:', repr(r.text))
    print('json:', r.json())
except Exception as e:
    print('error', e)
