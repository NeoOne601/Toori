#!/usr/bin/env python3
"""Quick runtime sanity check using a generated image and real API calls."""

import base64
import io
import json
import sys
import urllib.request

from PIL import Image


def post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def make_test_image(color):
    image = Image.new("RGB", (32, 32), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def main():
    try:
        analyze_one = post(
            "http://127.0.0.1:7777/v1/analyze",
            {"image_base64": make_test_image((255, 120, 0)), "session_id": "e2e", "decode_mode": "off"},
        )
        analyze_two = post(
            "http://127.0.0.1:7777/v1/analyze",
            {"image_base64": make_test_image((250, 130, 5)), "session_id": "e2e", "decode_mode": "off"},
        )
        search = post(
            "http://127.0.0.1:7777/v1/query",
            {"query": "red", "session_id": "e2e", "top_k": 5},
        )
    except Exception as exc:
        print("Runtime sanity check failed:", exc)
        sys.exit(1)

    print("First analyze response:", analyze_one)
    print("Second analyze response:", analyze_two)
    print("Query response:", search)


if __name__ == "__main__":
    main()
