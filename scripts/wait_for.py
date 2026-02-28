#!/usr/bin/env python3
"""Simple wait-for script to poll HTTP endpoints until available.

Usage: python scripts/wait_for.py http://minio:9000/minio/health/live http://mlflow:5000/
"""
import sys
import time
from urllib.request import urlopen, Request


def wait(url, timeout=60, interval=2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = Request(url, headers={"User-Agent": "wait-for/1.0"})
            with urlopen(req, timeout=5) as resp:
                if resp.status < 400:
                    print(f"OK: {url} -> {resp.status}")
                    return True
        except Exception as e:
            print(f"Waiting for {url}: {e}")
        time.sleep(interval)
    print(f"Timeout waiting for {url}")
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: wait_for.py <url> [<url> ...]")
        sys.exit(2)
    for u in sys.argv[1:]:
        ok = wait(u)
        if not ok:
            sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
