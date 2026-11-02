"""Regression checks for real-world dirty sales observations."""
import math

import numpy as np
import pandas as pd

from food_intelligence.elasticity import estimate_elasticities


def test_elasticity_ignores_nonpositive_nonfinite_and_non_numeric_sales():
    clean = [
        {"product_id": "A", "unit_price": 10 + i, "quantity": 100 - 3 * i}
        for i in range(12)
    ]
    malformed = [
        {"product_id": "A", "unit_price": 0, "quantity": 4},
        {"product_id": "A", "unit_price": -1, "quantity": 4},
        {"product_id": "A", "unit_price": math.inf, "quantity": 4},
        {"product_id": "A", "unit_price": 10, "quantity": np.nan},
        {"product_id": "A", "unit_price": "not-a-number", "quantity": 4},
    ]

    result = estimate_elasticities(pd.DataFrame(clean + malformed))
    reference = estimate_elasticities(pd.DataFrame(clean))

    assert result.loc["A", "n_obs"] == 12
    assert math.isfinite(result.loc["A", "elasticity"])
    assert result.loc["A", "elasticity"] < 0
    pd.testing.assert_frame_equal(result, reference)


def test_elasticity_skips_constant_prices_and_insufficient_clean_rows():
    constant = [
        {"product_id": "B", "unit_price": 10, "quantity": 4 + i}
        for i in range(12)
    ]
    too_few = [
        {"product_id": "C", "unit_price": 10 + i, "quantity": 30 - i}
        for i in range(9)
    ]
    too_few.append({"product_id": "C", "unit_price": 0, "quantity": 100})
    result = estimate_elasticities(pd.DataFrame(constant + too_few))
    assert result.empty
    assert list(result.columns) == ["elasticity", "r2", "n_obs"]


def test_elasticity_counts_only_valid_observations_for_each_product():
    rows = [
        {"product_id": "A", "unit_price": 8 + i, "quantity": 100 - 2 * i}
        for i in range(10)
    ]
    rows.append({"product_id": "A", "unit_price": None, "quantity": 1})
    rows += [
        {"product_id": "B", "unit_price": 9 + i, "quantity": 200 - 5 * i}
        for i in range(10)
    ]
    result = estimate_elasticities(pd.DataFrame(rows))
    assert list(result.index) == ["A", "B"]
    assert result.loc["A", "n_obs"] == 10
    assert result.loc["B", "n_obs"] == 10
