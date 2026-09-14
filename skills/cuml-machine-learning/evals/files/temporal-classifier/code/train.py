# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starter direct-cuML classifier with intentional leakage and key loss."""

import cudf
import pandas as pd
from cuml.linear_model import LogisticRegression
from cuml.preprocessing import StandardScaler

from split_contract import split_rows

FEATURES = ["balance", "payments_30d", "days_since_contact"]


def fit_and_score(examples, origin):
    """Fit on eligible history and return keyed scores for ``origin``."""
    # BUG: this learns scaling from training, scoring, and future rows.
    X_all = StandardScaler().fit_transform(
        examples[FEATURES].astype("float32")
    )
    prepared = examples.copy()
    prepared[FEATURES] = X_all

    training, scoring = split_rows(prepared, origin)
    model = LogisticRegression()
    model.fit(training[FEATURES], training["target"])

    # BUG: stable account/origin keys are discarded.
    return model.predict_proba(scoring[FEATURES])[:, 1]


def main():
    origin = pd.Timestamp("2026-04-01")
    examples = cudf.read_parquet("customer_examples.parquet")
    result = fit_and_score(examples, origin)
    print(result)


if __name__ == "__main__":
    main()
