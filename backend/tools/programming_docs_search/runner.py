#!/usr/bin/env python3
"""Entry point for the programming_docs_search tool."""

import base64
import json
import sys

from handler import lookup


def main():
    payload = json.loads(base64.b64decode(sys.argv[1]).decode())
    params = payload.get("params", {})

    language = (params.get("language") or "").strip()
    query = (params.get("query") or "").strip()

    if not language:
        print(json.dumps({"error": "Missing required parameter: language"}))
        sys.exit(1)
    if not query:
        print(json.dumps({"error": "Missing required parameter: query"}))
        sys.exit(1)

    result = lookup(language, query)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
