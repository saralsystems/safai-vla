"""Scripted expert policies for sewer robot sub-tasks."""

from policies.base import BasePolicy
from policies.deposit import DepositPolicy
from policies.extract import ExtractPolicy
from policies.navigate import NavigatePolicy
from policies.position import PositionPolicy

POLICY_MAP = {
    "navigate to blockage": NavigatePolicy,
    "assess and position for extraction": PositionPolicy,
    "extract sludge at current position": ExtractPolicy,
    "deposit extracted material": DepositPolicy,
}

__all__ = [
    "BasePolicy",
    "NavigatePolicy",
    "PositionPolicy",
    "ExtractPolicy",
    "DepositPolicy",
    "POLICY_MAP",
]
