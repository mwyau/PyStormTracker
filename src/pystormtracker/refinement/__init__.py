"""Feature-point refinement operations for regular and spherical grids."""

from __future__ import annotations

from .bspline import (
    BsplineRefinementResult,
    BsplineSurface,
    BsplineSurfaceResult,
    SphericalBsplineSurface,
    SphericalBsplineSurfaceResult,
    build_bspline_surface,
    build_spherical_bspline_surface,
    refine_bspline_feature_point,
    refine_spherical_bspline_feature_point,
)
from .quadratic import (
    SphericalQuadraticRefinementBatch,
    refine_quadratic_feature_point,
    refine_quadratic_feature_points,
    refine_spherical_quadratic_feature_points,
    spherical_quadratic_status_name,
)

__all__ = [
    "BsplineRefinementResult",
    "BsplineSurface",
    "BsplineSurfaceResult",
    "SphericalBsplineSurface",
    "SphericalBsplineSurfaceResult",
    "SphericalQuadraticRefinementBatch",
    "build_bspline_surface",
    "build_spherical_bspline_surface",
    "refine_bspline_feature_point",
    "refine_quadratic_feature_point",
    "refine_quadratic_feature_points",
    "refine_spherical_bspline_feature_point",
    "refine_spherical_quadratic_feature_points",
    "spherical_quadratic_status_name",
]
