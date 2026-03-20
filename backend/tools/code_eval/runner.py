import sys
import json
import base64
from handler import execute

payload = json.loads(base64.b64decode(sys.argv[1]))
params = payload.get("params", {})
settings = payload.get("settings", {})
telemetry = payload.get("telemetry", {})

result = execute(topic="", params=params, config=settings, telemetry=telemetry)

if result.get("error"):
    text = f"Error: {result['error']}"
    if result.get("output"):
        text = f"{result['output']}\n{text}"
else:
    text = result.get("output") or "(no output)"

print(json.dumps({"text": text}))
