import json, urllib.request, sys

data = json.dumps({"text":"天地","threshold":0.72,"max_retries":2}).encode('utf-8')
req = urllib.request.Request("http://127.0.0.1:8000/api/stroke_flow_run", data=data, headers={"Content-Type":"application/json"})
try:
    res = urllib.request.urlopen(req, timeout=15)
    print(res.read().decode('utf-8'))
except Exception as e:
    print("ERROR:", e, file=sys.stderr)
    sys.exit(1)
