# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starter split logic with intentional temporal leakage."""


def split_rows(examples, origin):
    """Return training and scoring rows for a prediction origin."""
    # BUG: a random row split ignores label availability and scoring origin.
    training = examples.sample(frac=0.75, random_state=42)
    scoring = examples.drop(training.index)
    return training, scoring
