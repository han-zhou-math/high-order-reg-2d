"""Configured problem objects shared by local, plotting, and all-grid drivers."""
import numpy as np

from .allgrid import LaplaceProblem, StokesProblem
from .config import KITE_BOX, STARFISH_BOX
from .geometries import (laplace_starfish_geometry, points_inside_curve,
                         starfish_level, stokes_kite_geometry)
from .manufactured_laplace import (exact_piecewise_laplace,
                                   laplace_boundary_densities)
from .manufactured_stokes import (exact_piecewise_stokes, grad_pressure,
                                  stokes_boundary_data)


def _kite_inside(points):
    return points_inside_curve(points, stokes_kite_geometry)


def _piecewise_pressure_gradient(points):
    inside = _kite_inside(points)
    return np.where(inside[..., None], grad_pressure(points), 0.0)


LAPLACE_STARFISH = LaplaceProblem(
    name="five_lobed_starfish",
    geometry=laplace_starfish_geometry,
    inside=lambda points: starfish_level(points) < 0,
    boundary_densities=laplace_boundary_densities,
    exact=exact_piecewise_laplace,
    box=STARFISH_BOX,
)

STOKES_KITE = StokesProblem(
    name="nonconvex_kite",
    geometry=stokes_kite_geometry,
    inside=_kite_inside,
    boundary_data=stokes_boundary_data,
    exact=exact_piecewise_stokes,
    exact_gradient=_piecewise_pressure_gradient,
    box=KITE_BOX,
)

