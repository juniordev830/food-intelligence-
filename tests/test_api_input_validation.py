import math

import pytest
from pydantic import ValidationError

from food_intelligence.api import ChurnRequest, ElasticityRequest, RecommendationRequest


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_churn_request_rejects_non_finite_money(bad_value):
    with pytest.raises(ValidationError):
        ChurnRequest(
            recency_days=1,
            frequency_90d=1,
            monetary_90d=bad_value,
            avg_order_value=10.0,
            n_categories=1,
            tenure_days=30,
        )


@pytest.mark.parametrize("field", ["current_price", "new_price", "current_quantity"])
@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_elasticity_request_rejects_non_finite_values(field, bad_value):
    values = {
        "product_id": "product-1",
        "current_price": 10.0,
        "new_price": 12.0,
        "current_quantity": 5.0,
    }
    values[field] = bad_value
    with pytest.raises(ValidationError):
        ElasticityRequest(**values)


def test_recommendation_request_rejects_too_many_products():
    with pytest.raises(ValidationError):
        RecommendationRequest(purchased_product_ids=[f"product-{i}" for i in range(501)])


def test_recommendation_request_accepts_maximum_product_count():
    request = RecommendationRequest(purchased_product_ids=[f"product-{i}" for i in range(500)])
    assert len(request.purchased_product_ids) == 500


def test_elasticity_request_rejects_empty_or_oversized_product_id():
    with pytest.raises(ValidationError):
        ElasticityRequest(product_id="", current_price=10.0, new_price=12.0, current_quantity=5.0)
    with pytest.raises(ValidationError):
        ElasticityRequest(product_id="x" * 101, current_price=10.0, new_price=12.0, current_quantity=5.0)


def test_recommendation_request_rejects_oversized_purchased_product_id():
    with pytest.raises(ValidationError):
        RecommendationRequest(purchased_product_ids=["x" * 101])


def test_recommendation_request_accepts_100_character_product_id():
    request = RecommendationRequest(purchased_product_ids=["x" * 100])
    assert len(request.purchased_product_ids[0]) == 100
