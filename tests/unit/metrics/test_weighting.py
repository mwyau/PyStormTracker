from __future__ import annotations

import numpy as np

from pystormtracker.metrics.weighting import _WeightType, calculate_spherical_weight


def test_constant_weight() -> None:
    # weight = 1.0 if d <= R else 0.0
    R = 500.0
    assert calculate_spherical_weight(0.0, R, int(_WeightType.CONSTANT), 20.0) == 1.0
    assert calculate_spherical_weight(250.0, R, int(_WeightType.CONSTANT), 20.0) == 1.0
    assert calculate_spherical_weight(500.0, R, int(_WeightType.CONSTANT), 20.0) == 1.0
    assert calculate_spherical_weight(500.1, R, int(_WeightType.CONSTANT), 20.0) == 0.0


def test_fisher_weight() -> None:
    # weight = exp(kappa * (cos(dist/Re) - 1))
    # At dist=0, weight should be 1.0
    assert np.allclose(
        calculate_spherical_weight(0.0, 500.0, int(_WeightType.FISHER), 20.0), 1.0
    )
    # Check decay
    w1 = calculate_spherical_weight(100.0, 500.0, int(_WeightType.FISHER), 20.0)
    w2 = calculate_spherical_weight(200.0, 500.0, int(_WeightType.FISHER), 20.0)
    assert 0.0 < w2 < w1 < 1.0


def test_cressman_weight() -> None:
    # weight = (R^2 - d^2) / (R^2 + d^2)
    R = 1000.0
    assert calculate_spherical_weight(0.0, R, int(_WeightType.CRESSMAN), 20.0) == 1.0
    # At d=R, weight should be 0.0
    assert calculate_spherical_weight(R, R, int(_WeightType.CRESSMAN), 20.0) == 0.0
    # At d=R/2 (500km), weight = (1 - 0.25) / (1 + 0.25) = 0.75 / 1.25 = 0.6
    assert np.allclose(
        calculate_spherical_weight(500.0, R, int(_WeightType.CRESSMAN), 20.0), 0.6
    )


def test_linear_weight() -> None:
    # weight = 1 - d/R
    R = 1000.0
    assert calculate_spherical_weight(0.0, R, int(_WeightType.LINEAR), 20.0) == 1.0
    assert calculate_spherical_weight(R, R, int(_WeightType.LINEAR), 20.0) == 0.0
    assert calculate_spherical_weight(500.0, R, int(_WeightType.LINEAR), 20.0) == 0.5


def test_quadratic_weight() -> None:
    # weight = 1 - (d/R)^2
    R = 1000.0
    assert calculate_spherical_weight(0.0, R, int(_WeightType.QUADRATIC), 20.0) == 1.0
    assert calculate_spherical_weight(R, R, int(_WeightType.QUADRATIC), 20.0) == 0.0
    # At d=R/2, weight = 1 - 0.25 = 0.75
    assert (
        calculate_spherical_weight(500.0, R, int(_WeightType.QUADRATIC), 20.0) == 0.75
    )
