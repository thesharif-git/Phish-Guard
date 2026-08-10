import urllib.request
import urllib.parse

data = urllib.parse.urlencode({'username': 'admin@phishguard.io', 'password': 'password'}).encode()
req = urllib.request.Request('http://127.0.0.1:8000/auth/login', data=data)

try:
    urllib.request.urlopen(req)
    print("Success")
except Exception as e:
    print(e.read().decode())
