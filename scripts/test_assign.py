import requests
r = requests.post('http://127.0.0.1:8080/assign',
                  json={'customer_id': 'test-1', 'ratio': 0.5}, timeout=5)
print('status', r.status_code)
print('text:', repr(r.text))
try:
    print('json:', r.json())
except Exception as e:
    print('json error:', e)
