"""Compatibility layer for classical forward difference routines."""

from black_scholes import (
    Gamma_forward,
    bs_price_call,
    delta_forward,
    gamma_forward,
)

__all__ = [
    "bs_price_call",
    "delta_forward",
    "gamma_forward",
    "Gamma_forward",
]

