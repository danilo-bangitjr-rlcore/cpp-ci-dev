from typing import Any


def pytest_addoption(parser: Any):
    parser.addoption("--feature_flag", action="store", default="base")
