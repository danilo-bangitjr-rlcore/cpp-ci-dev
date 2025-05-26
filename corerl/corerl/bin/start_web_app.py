#!/usr/bin/env python3

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for the webapp")
    parser.add_argument("--port", type=int, default=8000, help="Port for the webapp")
    args = parser.parse_args()

    uvicorn.run("corerl.web.app:app", host=args.host, port=args.port, timeout_graceful_shutdown=8)


if __name__ == "__main__":
    main()
