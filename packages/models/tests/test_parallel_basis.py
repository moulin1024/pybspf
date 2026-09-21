"""The setup worker pool must preserve every high-precision point row."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import BSpline
from bspf_models._numerics._weak_basis import mp_trial_values
from bspf_models._numerics._weak_basis import evaluate_mp_chunks


def test_parallel_high_precision_rows_are_identical():
    pytest.importorskip("gmpy2")
    rng = np.random.default_rng(3)
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context("spawn")) as pool:
        for n in (10,11):
            x=np.linspace(-1,1,n)
            knots=np.r_[np.repeat(-1.,4),np.linspace(-1,1,6)[1:-1],np.repeat(1.,4)]
            spline=BSpline(knots,np.eye(8),3)
            line=SimpleNamespace(x=x,P=rng.normal(size=(8,n)))
            points=np.r_[x,np.linspace(-1,1,257),x[:-1]+1e-14]
            options=dict(second=True,transform=rng.normal(size=(n+2,7)),layers=(0.03,))
            serial=mp_trial_values(line,spline,points,**options)
            parallel=evaluate_mp_chunks(line,spline,points,executor=pool,**options)
            for a,b in zip(serial,parallel):
                np.testing.assert_array_equal(a,b)

            (values,) = mp_trial_values(
                line, spline, points, values_only=True,
                **{key: value for key, value in options.items() if key != "second"},
            )
            np.testing.assert_array_equal(values, serial[0])
