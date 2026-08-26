from __future__ import annotations

import numpy as np
import pytest

from pystormtracker.metrics.weighting import _WeightType, calculate_spherical_weight


def test_constant_weight() -> None:
    # weight = 1.0 if d <= R else 0.0
    R = 500.0
    assert calculate_spherical_weight(0.0, R, int(_WeightType.CONSTANT)) == 1.0
    assert calculate_spherical_weight(250.0, R, int(_WeightType.CONSTANT)) == 1.0
    assert calculate_spherical_weight(500.0, R, int(_WeightType.CONSTANT)) == 1.0
    assert calculate_spherical_weight(500.1, R, int(_WeightType.CONSTANT)) == 0.0


def test_cressman_weight() -> None:
    # weight = (R^2 - d^2) / (R^2 + d^2)
    R = 1000.0
    assert calculate_spherical_weight(0.0, R, int(_WeightType.CRESSMAN)) == 1.0
    # At d=R, weight should be 0.0
    assert calculate_spherical_weight(R, R, int(_WeightType.CRESSMAN)) == 0.0
    # At d=R/2 (500km), weight = (1 - 0.25) / (1 + 0.25) = 0.75 / 1.25 = 0.6
    assert np.allclose(
        calculate_spherical_weight(500.0, R, int(_WeightType.CRESSMAN)), 0.6
    )


def test_linear_weight() -> None:
    # weight = 1 - d/R
    R = 1000.0
    assert calculate_spherical_weight(0.0, R, int(_WeightType.LINEAR)) == 1.0
    assert calculate_spherical_weight(R, R, int(_WeightType.LINEAR)) == 0.0
    assert calculate_spherical_weight(500.0, R, int(_WeightType.LINEAR)) == 0.5


def test_quadratic_weight() -> None:
    # weight = 1 - (d/R)^2
    R = 1000.0
    assert calculate_spherical_weight(0.0, R, int(_WeightType.QUADRATIC)) == 1.0
    assert calculate_spherical_weight(R, R, int(_WeightType.QUADRATIC)) == 0.0
    # At d=R/2, weight = 1 - 0.25 = 0.75
    assert calculate_spherical_weight(500.0, R, int(_WeightType.QUADRATIC)) == 0.75


@pytest.mark.parametrize(
    "weight_type",
    [_WeightType.CRESSMAN, _WeightType.LINEAR, _WeightType.QUADRATIC],
)
def test_compact_weights_are_zero_outside_their_support(
    weight_type: _WeightType,
) -> None:
    assert calculate_spherical_weight(1000.1, 1000.0, int(weight_type)) == 0.0
