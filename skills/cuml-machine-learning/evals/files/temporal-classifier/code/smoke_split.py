# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU semantic test for the backend-compatible temporal split contract."""

import pandas as pd

from split_contract import split_rows


def main():
    origin = pd.Timestamp("2026-04-01")
    examples = pd.DataFrame(
        {
            "account_id": [1, 2, 1, 2, 3],
            "prediction_origin": pd.to_datetime(
                [
                    "2026-03-01",
                    "2026-03-01",
                    "2026-04-01",
                    "2026-04-01",
                    "2026-05-01",
                ]
            ),
            "label_available_at": pd.to_datetime(
                ["2026-03-15", "2026-04-02", None, None, None]
            ),
            "target": [0, 1, None, None, None],
        }
    )

    training, scoring = split_rows(examples, origin)

    assert training["label_available_at"].notna().all()
    assert (training["label_available_at"] < origin).all()
    assert training["target"].notna().all()
    assert (scoring["prediction_origin"] == origin).all()
    assert set(scoring["account_id"]) == {1, 2}
    assert 3 not in set(scoring["account_id"])
    print("temporal split smoke test passed")


if __name__ == "__main__":
    main()
