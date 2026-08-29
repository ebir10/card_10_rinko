"""カードから読み取った数字を1個ずつちょうど使い切り、四則演算 (+ - * /) と
括弧で目標値(既定10)を作る式を総当たりで探すモジュール。

浮動小数点誤差を避けるため、内部の計算はすべて fractions.Fraction で行う。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations


@dataclass(frozen=True)
class _Node:
    value: Fraction
    text: str
    atomic: bool  # 単一の数字そのものなら True (括弧不要)

    def wrapped(self) -> str:
        return self.text if self.atomic else f"({self.text})"


def _combine(a: _Node, b: _Node) -> list[_Node]:
    results = [
        _Node(a.value + b.value, f"{a.wrapped()} + {b.wrapped()}", False),
        _Node(a.value * b.value, f"{a.wrapped()} * {b.wrapped()}", False),
        _Node(a.value - b.value, f"{a.wrapped()} - {b.wrapped()}", False),
        _Node(b.value - a.value, f"{b.wrapped()} - {a.wrapped()}", False),
    ]
    if b.value != 0:
        results.append(_Node(a.value / b.value, f"{a.wrapped()} / {b.wrapped()}", False))
    if a.value != 0:
        results.append(_Node(b.value / a.value, f"{b.wrapped()} / {a.wrapped()}", False))
    return results


def _search(nodes: list[_Node], target: Fraction) -> str | None:
    if len(nodes) == 1:
        return nodes[0].text if nodes[0].value == target else None
    n = len(nodes)
    for i, j in combinations(range(n), 2):
        rest = [nodes[k] for k in range(n) if k not in (i, j)]
        for combined in _combine(nodes[i], nodes[j]):
            found = _search(rest + [combined], target)
            if found is not None:
                return found
    return None


def solve(numbers: list[int], target: int = 10) -> str | None:
    """numbers を1個ずつちょうど使い切って target を作る式を返す。見つからなければ None。"""
    if len(numbers) > 7:
        raise ValueError("計算量が爆発するため7枚以下にしてください")
    nodes = [_Node(Fraction(v), str(v), True) for v in numbers]
    return _search(nodes, Fraction(target))
