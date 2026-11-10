# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select resolved workloads using a small, pytest-like keyword language."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, Literal, NoReturn

from .suite import ResolvedCase, SuiteError, SuitePlan

type Node = (
    tuple[Literal["term"], str]
    | tuple[Literal["not"], Node]
    | tuple[Literal["and", "or"], list[Node]]
)


def case_keywords(case: ResolvedCase) -> tuple[str, ...]:
    """Return searchable and resolved workload descriptors."""
    dtype = case.dtypes["y" if case.input_selection == ("y",) else "X"]
    return (
        case.estimator,
        case.operation,
        f"{case.estimator}.{case.operation}",
        case.dataset,
        case.label,
        case.workload_id(),
        f"rows={case.measured_rows}",
        f"features={case.features}",
        f"dtype={dtype}",
    )


def compile_expression(expression: str) -> Callable[[ResolvedCase], bool]:
    """Compile a keyword expression for resolved benchmark cases."""
    matches = compile_keyword_expression(expression)
    return lambda case: matches(case_keywords(case))


def compile_keyword_expression(
    expression: str,
) -> Callable[[tuple[str, ...]], bool]:
    """Parse substring terms, parentheses, and lowercase not/and/or operators."""
    tokens = re.findall(r"\(|\)|[^\s()]+", expression)
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def fail() -> NoReturn:
        token = peek()
        raise SuiteError(
            f"invalid -k expression {expression!r} at token {position + 1}: "
            + (
                f"unexpected {token!r}"
                if token is not None
                else "unexpected end"
            )
        )

    def atom() -> Node:
        nonlocal position
        token = peek()
        if token == "(":
            position += 1
            node = disjunction()
            if peek() != ")":
                fail()
            position += 1
            return node
        if token is None or token in {"and", "or", "not", ")"}:
            fail()
        position += 1
        return ("term", token.casefold())

    def negation() -> Node:
        nonlocal position
        # Iterative to support repeated negation without recursive nesting.
        count = 0
        while peek() == "not":
            position += 1
            count += 1
        node = atom()
        return ("not", node) if count % 2 else node

    def conjunction() -> Node:
        nonlocal position
        nodes = [negation()]
        while peek() == "and":
            position += 1
            nodes.append(negation())
        return ("and", nodes)

    def disjunction() -> Node:
        nonlocal position
        nodes = [conjunction()]
        while peek() == "or":
            position += 1
            nodes.append(conjunction())
        return ("or", nodes)

    if not tokens:
        return lambda keywords: True
    try:
        tree = disjunction()
    except RecursionError as exc:
        raise SuiteError("-k expression is nested too deeply") from exc
    if peek() is not None:
        fail()

    def evaluate(node: Node, keywords: tuple[str, ...]) -> bool:
        if node[0] == "term":
            return any(node[1] in keyword for keyword in keywords)
        if node[0] == "not":
            return not evaluate(node[1], keywords)
        values = (evaluate(child, keywords) for child in node[1])
        return all(values) if node[0] == "and" else any(values)

    def matches(keywords: tuple[str, ...]) -> bool:
        return evaluate(tree, tuple(k.casefold() for k in keywords))

    return matches


def select_plan(plan: SuitePlan, expression: str) -> SuitePlan:
    """Filter a validated plan, preserving workload identity and manifest order."""
    matches = compile_expression(expression)
    runs = []
    for run in plan.runs:
        cases = tuple(case for case in run.cases if matches(case))
        if cases:
            runs.append(replace(run, cases=cases))
    if not runs:
        raise SuiteError(f"no benchmark cases match -k {expression!r}")
    return replace(plan, runs=tuple(runs), keyword_expression=expression)
