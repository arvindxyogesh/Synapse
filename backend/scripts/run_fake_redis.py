#!/usr/bin/env python3
"""Stand-in Redis server for environments where installing a real
redis-server/valkey-server binary isn't possible (no root access, and no
matching package in the available conda/apt channels -- e.g. some shared
HPC clusters).

Runs fakeredis's pure-Python TCP server, which speaks the real Redis wire
protocol -- Synapse's redis-py client works against it completely
unmodified. This is fine for local dev and benchmarking (nothing about
cache correctness, latency, or cost depends on which Redis-compatible
server is behind it); production should run a real Redis or Valkey
instance (see docker-compose.yml).
"""

import argparse

from fakeredis import TcpFakeServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    args = parser.parse_args()

    server = TcpFakeServer((args.host, args.port))
    print(f"fake redis listening on {args.host}:{args.port} (ctrl-c to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
