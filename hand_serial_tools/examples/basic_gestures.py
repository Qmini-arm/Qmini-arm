#!/usr/bin/env python3
"""Small five-finger API example; no wrist or arm joint is addressed."""

from __future__ import annotations

import argparse

from uhand import connect


def main() -> None:
    parser = argparse.ArgumentParser(description="uHand five-finger gesture demo")
    parser.add_argument("--port", required=True)
    args = parser.parse_args()

    with connect(args.port) as hand:
        hand.gesture("open", duration=0.6)
        hand.gesture("point", duration=0.6)
        hand.gesture("victory", duration=0.6)
        hand.gesture("open", duration=0.6)


if __name__ == "__main__":
    main()
