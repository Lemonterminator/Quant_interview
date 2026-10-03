"""Validated NumPy batch facade over the existing call-pricing functions.

This is the stable comparison interface for scalar, broadcast, and C++ runs.
It intentionally leaves the assignment functions in ``black_scholes.py`` intact.
NumPy vectorization means array-wise native loops; it does not promise multicore
execution. See ``benchmarks/benchmark_python.py`` for an explicit thread mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from black_scholes import Gamma, _validate_cdf_order, bs_price_call, delta_call, delta_cs, gamma_45_deg


@dataclass(frozen=True)
class CallResult:
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray


def evaluate_call(S, K, r, q, sigma, T) -> CallResult:
    """Evaluate a European call on broadcast-compatible float64 inputs.

    Contract: finite S,K>0, finite r,q, and finite sigma,T>=0. Outputs retain
    the broadcast shape, including shape ``()`` for scalar inputs. At sigma=0
    or T=0, price is discounted forward intrinsic value and gamma is set to 0.
    Delta at the intrinsic kink uses the explicit midpoint convention 0.5.
    This convention is a library choice, not a derivative at the kink.
    """

    values = np.broadcast_arrays(
        *(np.asarray(value, dtype=np.float64) for value in (S, K, r, q, sigma, T))
    )
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("All Black-Scholes inputs must be finite")
    spot, strike, rate, yield_, vol, expiry = values
    if np.any(spot <= 0) or np.any(strike <= 0) or np.any(vol < 0) or np.any(expiry < 0):
        raise ValueError("Require S,K>0 and sigma,T>=0")

    shape = spot.shape
    flat = tuple(np.ravel(value) for value in values)
    sf, kf, rf, qf, vf, tf = flat
    price = np.empty(sf.size, dtype=np.float64)
    delta = np.empty_like(price)
    gamma = np.empty_like(price)
    regular = (vf > 0) & (tf > 0)

    if np.any(regular):
        args = tuple(value[regular] for value in flat)
        price[regular] = bs_price_call(*args)
        delta[regular] = delta_call(*args)
        gamma[regular] = Gamma(*args)

    degenerate = ~regular
    if np.any(degenerate):
        a = sf[degenerate] * np.exp(-qf[degenerate] * tf[degenerate])
        b = kf[degenerate] * np.exp(-rf[degenerate] * tf[degenerate])
        price[degenerate] = np.maximum(a - b, 0.0)
        delta[degenerate] = np.exp(-qf[degenerate] * tf[degenerate]) * (
            (a > b).astype(np.float64) + 0.5 * (a == b).astype(np.float64)
        )
        gamma[degenerate] = 0.0

    if not (np.all(np.isfinite(price)) and np.all(np.isfinite(delta)) and np.all(np.isfinite(gamma))):
        raise OverflowError("Black-Scholes result is not representable as float64")
    return CallResult(price.reshape(shape), delta.reshape(shape), gamma.reshape(shape))


def evaluate_call_complex_step(S, K, r, q, sigma, T, *, h=1e-5, order=3) -> CallResult:
    """Validated real-input batch with CS Delta and 45-degree CS Gamma.

    Price uses the analytic real pricer. For sigma=0 or T=0, use the analytic
    facade's midpoint/zero-Gamma convention. Positive finite absolute spot
    steps h broadcast to the input result shape. The caller must select a
    numerically resolvable h for Gamma; CDF order does not select the step.
    The existing complex pricer also rejects sigma*sqrt(T) close to zero.
    """
    _validate_cdf_order(order)
    baseline = evaluate_call(S, K, r, q, sigma, T)
    values = np.broadcast_arrays(*(np.asarray(x, dtype=np.float64) for x in (S, K, r, q, sigma, T)))
    step = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(step)) or np.any(step <= 0):
        raise ValueError("h must contain positive finite absolute spot steps")
    step = np.broadcast_to(step, baseline.price.shape).ravel()
    flat = tuple(x.ravel() for x in values)
    regular = (flat[4] > 0) & (flat[5] > 0)
    delta, gamma = baseline.delta.copy().ravel(), baseline.gamma.copy().ravel()
    if np.any(regular):
        args = tuple(x[regular] for x in flat)
        delta[regular] = delta_cs(*args, h=step[regular], order=order)
        gamma[regular] = gamma_45_deg(*args, h=step[regular], order=order)
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(gamma)):
        raise OverflowError("Complex-step Greeks are not representable as float64")
    return CallResult(baseline.price, delta.reshape(baseline.price.shape), gamma.reshape(baseline.price.shape))
