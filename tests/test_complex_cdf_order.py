"""Golden coefficients, public argument propagation and local error order."""
import csv
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Python_prototype'))
sys.path.insert(0, str(ROOT))
import black_scholes as bs
from bs_batch import evaluate_call, evaluate_call_complex_step
import complex_step_differentiation as legacy


class ComplexCdfOrderTests(unittest.TestCase):
    def test_golden_coefficients_all_orders(self):
        with (ROOT / 'testdata/complex_cdf_golden.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 20)
        for order in range(2, 7):
            chosen = [x for x in rows if int(x['order']) == order]
            z = np.array([complex(float(x['a']), float(x['b'])) for x in chosen])
            expected = np.array([complex(float(x['real']), float(x['imag'])) for x in chosen])
            actual = bs.Phi_t(z, order=order)
            np.testing.assert_allclose(actual.real, expected.real, rtol=2e-15, atol=2e-16)
            np.testing.assert_allclose(actual.imag, expected.imag, rtol=2e-15, atol=1e-18)

    def test_default_p3_and_legacy_exports(self):
        args = (100., 100., 0., 0., .2, 1.)
        z = .3 + .04j
        self.assertEqual(bs.Phi_t(z), bs.Phi_t(z, order=3))
        for fn in (bs.delta_cs, bs.gamma_cs_real, bs.gamma_45_deg):
            self.assertEqual(fn(*args, h=.2), fn(*args, h=.2, order=3))
        self.assertEqual(bs.bs_price_call_t(100+.2j, *args[1:]), bs.bs_price_call_t(100+.2j, *args[1:], order=3))
        self.assertIs(legacy.Phi_t, bs.Phi_t)
        self.assertIs(legacy.gamma_45_deg, bs.gamma_45_deg)
        self.assertIs(bs.Phi_dispatch, bs.Phi_t)
        self.assertIs(bs.Gamma_45_deg, bs.gamma_45_deg)
        self.assertGreater(abs(bs.gamma_45_deg(*args, h=.2, order=2)-bs.gamma_45_deg(*args, h=.2)), 1e-7)

    def test_real_inputs_empty_arrays_and_np_integer(self):
        real = np.array([[-3., 0., .3]])
        for order in range(2, 7):
            np.testing.assert_array_equal(bs.Phi_t(real, order=order), bs.Phi_real(real))
            self.assertEqual(bs.Phi_t(np.array([], dtype=complex), order=order).shape, (0,))
        self.assertEqual(bs.Phi_t(.3j, order=np.int64(3)), bs.Phi_t(.3j, order=3))

    def test_sweep_forwarding_and_positional_compatibility(self):
        from experiments.assignment import black_scholes_sweep as sweep
        scenario = sweep.SCENARIOS[0]
        args = (scenario.S, scenario.K, scenario.r, scenario.q, scenario.sigma, scenario.T)
        for order in range(2, 7):
            with patch.object(sweep, 'gamma_45_deg', wraps=bs.gamma_45_deg) as gamma:
                sweep.run_gamma_sweep(scenario, np.array([.002]), order=order)
                self.assertTrue(all(c.kwargs['order'] == order for c in gamma.call_args_list))
                self.assertGreater(gamma.call_count, 0)
            with patch.object(Path, 'mkdir'):
                self.assertEqual(sweep.scenario_dir('example', order=order), sweep.RESULTS_DIR / f'p{order}' / 'example')
        for fn in (bs.delta_cs, bs.gamma_cs_real, bs.gamma_45_deg):
            self.assertEqual(fn(*args, .2), fn(*args, h=.2, order=3))
        self.assertEqual(bs.bs_price_call_t(100+.2j, *args[1:], 1e-12), bs.bs_price_call_t(100+.2j, *args[1:], eps=1e-12, order=3))

    def test_invalid_order_at_every_layer(self):
        args = (100., 100., 0., 0., .2, 1.)
        functions = [lambda order: bs.Phi_t(0., order=order),
                     lambda order: bs.bs_price_call_t(*args, order=order),
                     lambda order: bs.delta_cs(*args, order=order),
                     lambda order: bs.gamma_cs_real(*args, order=order),
                     lambda order: bs.gamma_45_deg(*args, order=order),
                     lambda order: evaluate_call_complex_step(*args, order=order)]
        for invalid in (1, 7, 3., '3', True, np.array([3]), None):
            for fn in functions:
                with self.subTest(order=repr(invalid)), self.assertRaises((ValueError, TypeError)):
                    fn(invalid)
        with self.assertRaises(ValueError):
            evaluate_call_complex_step(np.array([]),100,0,0,.2,1,order=7)

    def test_greeks_forward_order_through_price_to_cdf(self):
        args = (100., 100., 0., 0., .2, 1.)
        for fn in (bs.delta_cs, bs.gamma_cs_real, bs.gamma_45_deg):
            for order in range(2, 7):
                with patch.object(bs, 'Phi_t', wraps=bs.Phi_t) as cdf:
                    fn(*args, h=.2, order=order)
                    self.assertGreaterEqual(cdf.call_count, 2)
                    self.assertTrue(all(call.kwargs['order'] == order for call in cdf.call_args_list))

    def test_local_second_vs_fourth_order(self):
        x = .3
        omega = (1+1j)/np.sqrt(2)
        truth = -x*float(bs.phi(x))
        for order in range(2, 7):
            errors = []
            for h in (.04, .02):
                actual = np.imag(bs.Phi_t(x+omega*h,order=order)+bs.Phi_t(x-omega*h,order=order))/h**2
                errors.append(abs(actual-truth))
            target = 4 if order==2 else 16
            self.assertLess(abs(errors[0]/errors[1]-target), .1)

    def test_batch_scalar_broadcast_steps_and_degenerate(self):
        inputs = (np.array([[90.], [110.]]), np.array([[80., 100., 120.]]), .01, 0., .2, 1.)
        steps = np.array([[.1, .2, .3]])
        for order in range(2, 7):
            result = evaluate_call_complex_step(*inputs, h=steps, order=order)
            self.assertEqual(result.gamma.shape, (2,3))
            for i,s in enumerate((90.,110.)):
                for j,k in enumerate((80.,100.,120.)):
                    self.assertAlmostEqual(result.gamma[i,j],bs.gamma_45_deg(s,k,.01,0.,.2,1.,h=steps[0,j],order=order),places=12)
            degenerate = evaluate_call_complex_step([90,100,110],100,0,0,0,1,order=order)
            expected = evaluate_call([90,100,110],100,0,0,0,1)
            for field in ('price','delta','gamma'):
                np.testing.assert_array_equal(getattr(degenerate,field),getattr(expected,field))
        self.assertEqual(evaluate_call_complex_step(np.array([]),100,0,0,.2,1).gamma.size,0)
        scalar = evaluate_call_complex_step(100,100,0,0,.2,1,h=.2)
        self.assertEqual(scalar.gamma.shape,())
        for h in (0,-1,np.nan,np.inf):
            with self.assertRaises(ValueError): evaluate_call_complex_step(100,100,0,0,.2,1,h=h)
        with self.assertRaises(ValueError): evaluate_call_complex_step(*inputs,h=np.ones(4))


if __name__=='__main__':
    unittest.main()
