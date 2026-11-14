"""Authoritative geometry, field, refinement, and output configuration."""
from pathlib import Path

ORDERS = (3, 5, 7)
ORDER_TO_M = {3: 1, 5: 2, 7: 3}
ORDER_TO_Q = {3: 2 / 3, 5: 4 / 5, 7: 5 / 7}
KAPPA = 1.0
LOCAL_LAMBDAS = (0.0, 0.5)
ROBUSTNESS_LAMBDAS = (0.0, 0.5, 1.0, 2.0)
BOUNDARY_LEVELS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
CARTESIAN_LEVELS = (64, 128, 256, 512, 768, 1024)
TARGET_COUNT = 97
OVERSAMPLE_FACTOR = 2
BOUNDARY_TO_CARTESIAN_RATIO = 8

STARFISH_BOX = (-1.6, 1.6, -1.6, 1.6)
KITE_BOX = (-2.2, 1.6, -1.9, 1.9)

RESULT_ROOT = Path("results/redesigned")
FIGURE_ROOT = Path("figures")

# Geometry and manufactured-field constants have one authoritative definition
# here; exact-solution and plotting modules import them.
STARFISH_BASE_RADIUS = 1.0
STARFISH_AMPLITUDE = 0.3
STARFISH_LOBES = 5
KITE_SECOND_HARMONIC = 0.65
KITE_VERTICAL_SCALE = 1.5

CIRCLE_COS3 = 0.4
CIRCLE_SIN5 = 0.3

STARFISH_FMINUS_POLES = (1.65 + 0.15j, -1.45 + 0.65j)
STARFISH_FPLUS_POLES = (0.18 + 0.12j, -0.22 + 0.08j, 0.05 - 0.25j)
STARFISH_FMINUS_COEFFICIENTS = {
    "exponential": 0.35, "exponent": 0.30, "z4": 0.08, "z3": 0.12j,
    "residues": (0.25, -0.16j),
}
STARFISH_FPLUS_RESIDUES = (0.40 + 0.08j, -0.18j, 0.08)

KITE_F_POLES = (2.10 + 0.40j, -2.20 + 0.50j)
KITE_G_POLES = (0.40 + 2.30j, -0.60 - 2.20j)
KITE_F_COEFFICIENTS = {
    "z4": 0.10, "z2": 0.06 + 0.04j, "exponential": 0.12,
    "exponent": 0.25, "residues": (0.15, -0.10j),
}
KITE_G_COEFFICIENTS = {
    "z5": -0.06j, "z3": 0.09, "exponential": 0.08,
    "exponent": -0.20j, "residues": (0.10, 0.07j),
}
