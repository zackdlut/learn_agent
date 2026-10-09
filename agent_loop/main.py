#!/usr/bin/env python3

from .agent import Agent
from .env import load_env
from .http_logger import clear_log


def main():
    load_env()
    clear_log()
    Agent().run()


if __name__ == "__main__":
    main()
