# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from deel.puncc.api.conformalization import ConformalPredictor
from deel.puncc.api.conformalization import CrossValCpAggregator
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.classification import LAC
from deel.puncc.classification import RAPS
from deel.puncc.plotting import draw_bounding_box
from deel.puncc.plotting import plot_prediction_intervals
from deel.puncc.regression import CVPlus
from deel.puncc.regression import EnbPI
from deel.puncc.regression import SplitCP


class DummyConformalPredictor:
    def __init__(self, prediction=None):
        self.splitter = None
        self.prediction = prediction if prediction is not None else (np.array([1.0]), np.array([0.0]), np.array([2.0]))
        self.nconf_scores = {0: np.array([1.0, 2.0])}

    def fit(self, X=None, y=None, **kwargs):
        self.fit_args = (X, y, kwargs)

    def predict(self, X, alpha):
        del X, alpha
        return self.prediction

    def get_nonconformity_scores(self):
        return self.nconf_scores


class DummyPredictor:
    def __init__(self, trained=True):
        self.is_trained = trained

    def copy(self):
        return DummyPredictor(trained=self.is_trained)

    def fit(self, X, y, **kwargs):
        self.fit_args = (X, y, kwargs)

    def predict(self, X=None, **kwargs):
        del kwargs
        X = np.asarray(X)
        return np.ones(len(X))

    def get_is_trained(self):
        return self.is_trained


class BareModelWithFitPredict:
    def fit(self, X, y, **kwargs):
        self.fit_args = (X, y, kwargs)

    def predict(self, X):
        X = np.asarray(X)
        return np.ones(len(X))


class DummyCalibrator:
    def __init__(self, weight_func=None):
        self.weight_func = weight_func
        self._scores = np.array([1.0, 2.0])

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs

    def barber_weights(self, weights):
        return np.asarray(weights) / np.sum(weights)

    def set_norm_weights(self, weights):
        self._norm_weights = weights

    def get_nonconformity_scores(self):
        return self._scores

    def get_weights(self):
        return getattr(self, "_norm_weights", None)

    def get_norm_weights(self):
        return getattr(self, "_norm_weights", None)

    def calibrate(self, X, alpha, y_pred, weights=None, correction=None):
        del X, alpha, correction
        return (y_pred - 1, y_pred + 1)


class BarePredictModel:
    def predict(self, X):
        return np.asarray(X)


class NoFitPredictModel:
    def predict(self, X):
        return np.asarray(X)


def test_lac_wrapper_branches():
    lac = LAC(DummyPredictor())
    lac.conformal_predictor = DummyConformalPredictor(prediction=(np.array([0.8]), ([0],)))

    y_pred, set_pred = lac.predict(np.array([[1.0]]), alpha=0.1)
    np.testing.assert_allclose(y_pred, np.array([0.8]))
    assert set_pred == ([0],)

    lac.conformal_predictor = None
    with pytest.raises(AttributeError):
        lac.predict(np.array([[1.0]]), alpha=0.1)

    lac2 = LAC(DummyPredictor(trained=False), train=False)
    lac2.predictor.is_trained = True
    lac2.conformal_predictor = DummyConformalPredictor()
    lac2.fit(
        X_calib=np.array([[1.0]]),
        y_calib=np.array([1]),
    )
    assert lac2.conformal_predictor.fit_args[0] is None
    assert lac2.conformal_predictor.fit_args[1] is None

    lac3 = LAC(DummyPredictor(), train=True)
    lac3.conformal_predictor = DummyConformalPredictor()
    with pytest.raises(RuntimeError, match="Argument 'train' is True"):
        lac3.fit(
            X_calib=np.array([[1.0]]),
            y_calib=np.array([1]),
        )

    lac4 = LAC(DummyPredictor())
    lac4.conformal_predictor = DummyConformalPredictor()
    lac4.fit(X=np.array([[1.0], [2.0]]), y=np.array([0, 1]))
    assert lac4.conformal_predictor.splitter is not None

    lac5 = LAC(DummyPredictor())
    lac5.conformal_predictor = DummyConformalPredictor()
    lac5.fit(
        X_fit=np.array([[1.0]]),
        y_fit=np.array([1]),
        X_calib=np.array([[2.0]]),
        y_calib=np.array([0]),
    )
    assert lac5.conformal_predictor.splitter is not None

    lac6 = LAC(DummyPredictor())
    lac6.conformal_predictor = DummyConformalPredictor()
    with pytest.raises(RuntimeError, match="No dataset provided"):
        lac6.fit()

    lac7 = LAC(DummyPredictor())
    lac7.conformal_predictor = DummyConformalPredictor()
    np.testing.assert_allclose(lac7.get_nonconformity_scores(), np.array([1.0, 2.0]))
    lac7.conformal_predictor.nconf_scores = {
        0: np.array([1.0]),
        1: np.array([2.0]),
    }
    assert isinstance(lac7.get_nonconformity_scores(), dict)


def test_raps_wrapper_branches():
    raps = RAPS(DummyPredictor())
    raps.conformal_predictor = DummyConformalPredictor(prediction=(np.array([0.7]), ([0, 1],)))

    y_pred, set_pred = raps.predict(np.array([[1.0]]), alpha=0.1)
    np.testing.assert_allclose(y_pred, np.array([0.7]))
    assert set_pred == ([0, 1],)

    raps.conformal_predictor = None
    with pytest.raises(RuntimeError):
        raps.predict(np.array([[1.0]]), alpha=0.1)

    raps2 = RAPS(DummyPredictor())
    raps2.conformal_predictor = DummyConformalPredictor()
    raps2.fit(X=np.array([[1.0], [2.0]]), y=np.array([0, 1]))
    assert raps2.conformal_predictor.splitter is not None

    raps3 = RAPS(DummyPredictor())
    raps3.conformal_predictor = DummyConformalPredictor()
    raps3.fit(
        X_calib=np.array([[1.0]]),
        y_calib=np.array([1]),
    )
    assert raps3.conformal_predictor.splitter is not None

    raps4 = RAPS(DummyPredictor(trained=False), train=False)
    raps4.conformal_predictor = DummyConformalPredictor()
    with pytest.raises(RuntimeError, match="No dataset provided"):
        raps4.fit(
            X_calib=np.array([[1.0]]),
            y_calib=np.array([1]),
        )

    raps5 = RAPS(DummyPredictor())
    raps5.conformal_predictor = DummyConformalPredictor()
    with pytest.raises(RuntimeError, match="No dataset provided"):
        raps5.fit()


def test_splitcp_wrapper_branches():
    split_cp = SplitCP(DummyPredictor())
    split_cp.conformal_predictor = DummyConformalPredictor()
    y_pred, y_lo, y_hi = split_cp.predict(np.array([[1.0]]), alpha=0.1)
    np.testing.assert_allclose(y_pred, np.array([1.0]))
    np.testing.assert_allclose(y_lo, np.array([0.0]))
    np.testing.assert_allclose(y_hi, np.array([2.0]))

    del split_cp.conformal_predictor
    with pytest.raises(AttributeError):
        split_cp.predict(np.array([[1.0]]), alpha=0.1)


def test_enbpi_branch_methods():
    enbpi = EnbPI(DummyPredictor(), B=1)
    enbpi.residuals = [1.0]

    with pytest.raises(RuntimeError):
        enbpi.predict(np.array([[1.0]]), alpha=0.1)

    enbpi._boot_predictors = [DummyPredictor()]
    enbpi.residuals = [1.0, 2.0]
    enbpi._oob_matrix = np.array([[1.0]])
    y_pred, y_lo, y_hi = enbpi.predict(np.array([[1.0]]), alpha=0.1)
    assert len(y_pred) == len(y_lo) == len(y_hi) == 1

    enbpi._boot_predictors = None
    enbpi.residuals = [1.0]
    with pytest.raises(RuntimeError):
        enbpi.predict(np.array([[1.0]]), alpha=0.1, y_true=np.array([1.0]), s=1)

    enbpi._boot_predictors = [DummyPredictor()]
    enbpi.residuals = [1.0]
    enbpi._oob_matrix = np.array([[1.0]])
    enbpi.B = 1
    y_pred_scalar, _, _ = enbpi.predict(np.array([[1.0]]), alpha=0.1)
    assert len(y_pred_scalar) == 1


def test_plotting_branch_cases(tmp_path):
    ax = plot_prediction_intervals(
        y_true=np.array([1.0, 2.0]),
        y_pred=None,
        y_pred_lower=None,
        y_pred_upper=None,
        X=None,
    )
    assert ax.get_title() == ""
    plt.close(ax.figure)

    ax2 = plot_prediction_intervals(
        y_true=np.array([2.0, 2.0]),
        y_pred=np.array([2.0, 2.0]),
        y_pred_lower=np.array([1.5, 1.5]),
        y_pred_upper=np.array([2.5, 2.5]),
        X=np.array([1.0, 1.0]),
        loc="lower right",
    )
    xmin, xmax = ax2.get_xlim()
    assert xmin < 1.0 < xmax
    plt.close(ax2.figure)

    before_font = matplotlib.rcParams["font.family"]
    ax3 = plot_prediction_intervals(
        y_true=np.array([1.0, 3.0]),
        y_pred=np.array([1.5, 2.5]),
        y_pred_lower=np.array([0.5, 1.5]),
        y_pred_upper=np.array([2.5, 3.5]),
        figsize=(6, 4),
    )
    assert ax3.get_title() == "Prediction Intervals | coverage=1.000"
    assert matplotlib.rcParams["font.family"] == before_font
    plt.close(ax3.figure)

    image_path = tmp_path / "bbox.png"
    Image.new("RGB", (10, 10), color="white").save(image_path)
    im = draw_bounding_box(image_path=str(image_path), legend="loaded", show=False)
    assert im.size == (10, 10)

    image = Image.new("RGB", (10, 10), color="white")
    image.custom_lines = []
    image.legends = []
    shown = draw_bounding_box(image=image, show=True, legend="")
    assert shown.size == (10, 10)

    image_with_legend = Image.new("RGB", (10, 10), color="white")
    shown_with_legend = draw_bounding_box(
        image=image_with_legend,
        box=(1, 1, 5, 5),
        legend="box",
        show=True,
    )
    assert shown_with_legend.legends == ["box"]
    plt.close("all")


def test_conformalization_guard_and_aggregator_branches(tmp_path):
    with pytest.raises(RuntimeError):
        ConformalPredictor(
            calibrator=DummyCalibrator(),
            predictor=object(),
            splitter=object(),
        )

    with pytest.raises(RuntimeError):
        ConformalPredictor(
            calibrator=DummyCalibrator(),
            predictor=BarePredictModel(),
            splitter=object(),
            train=True,
        )

    wrapped = ConformalPredictor(
        calibrator=DummyCalibrator(),
        predictor=BareModelWithFitPredict(),
        splitter=object(),
        train=True,
    )
    assert isinstance(wrapped.predictor, BasePredictor)

    cp = ConformalPredictor(
        calibrator=DummyCalibrator(),
        predictor=BarePredictModel(),
        splitter=object(),
        train=False,
    )
    with pytest.raises(RuntimeError):
        ConformalPredictor(
            calibrator=DummyCalibrator(),
            predictor=DummyPredictor(),
            splitter=object(),
            method="bad",
        )

    with pytest.raises(RuntimeError):
        cp.get_nonconformity_scores()
    with pytest.raises(RuntimeError):
        cp.get_weights()
    with pytest.raises(RuntimeError):
        cp.predict(np.array([[1.0]]), alpha=0.1)

    cp.splitter = None
    cp.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
    assert cp.get_nonconformity_scores() is not None
    assert cp.get_weights() == {0: None}

    cp_untrained = ConformalPredictor(
        calibrator=DummyCalibrator(),
        predictor=DummyPredictor(trained=False),
        splitter=type("MultiSplitter", (), {"__call__": lambda self, X, y: [(X, y, X, y), (X, y, X, y)]})(),
        train=False,
    )
    with pytest.raises(RuntimeError):
        cp_untrained.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    cp_bad_splitter = ConformalPredictor(
        calibrator=DummyCalibrator(),
        predictor=DummyPredictor(),
        splitter=None,
        train=False,
    )
    cp_bad_splitter.train = True
    with pytest.raises(RuntimeError):
        cp_bad_splitter.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    weighted_cp = ConformalPredictor(
        calibrator=DummyCalibrator(weight_func=lambda X: np.arange(1, len(X) + 1)),
        predictor=DummyPredictor(),
        splitter=None,
        train=False,
    )
    weighted_cp.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(weighted_cp.get_weights()[0], np.array([1 / 3, 2 / 3]))

    cached_cp = ConformalPredictor(
        calibrator=DummyCalibrator(),
        predictor=DummyPredictor(),
        splitter=type(
            "SingleSplit",
            (),
            {"__call__": lambda self, X, y: [(X[:1], y[:1], X[1:], y[1:])]},
        )(),
        train=False,
    )
    cached_cp.predictor.is_trained = True
    cached_cp.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
    cached_cp.fit(np.array([[3.0], [4.0]]), np.array([3.0, 4.0]), use_cached=True)
    assert cached_cp._cv_cp_agg.K == 2

    path = tmp_path / "cp.pkl"
    cp.save(path, save_data=True)
    loaded = ConformalPredictor.load(path)
    assert isinstance(loaded, ConformalPredictor)

    agg = CrossValCpAggregator(K=1)
    dummy_predictor = DummyPredictor()
    dummy_calibrator = DummyCalibrator(weight_func=lambda X: np.array([0.5] * len(X)))
    dummy_calibrator.set_norm_weights(np.array([0.5, 0.5]))
    agg.append_predictor(0, dummy_predictor)
    agg.append_calibrator(0, dummy_calibrator)
    y_pred, y_lo, y_hi = agg.predict(np.array([[1.0], [2.0]]), alpha=0.1)
    assert y_pred.shape == y_lo.shape == y_hi.shape
    np.testing.assert_allclose(agg.get_weights()[0], np.array([0.5, 0.5]))

    bad_agg = CrossValCpAggregator(K=1)
    bad_agg._predictors = {0: dummy_predictor}
    bad_agg._calibrators = {1: dummy_calibrator}
    with pytest.raises(AssertionError):
        bad_agg.predict(np.array([[1.0]]), alpha=0.1)

    with pytest.raises(NotImplementedError):
        CrossValCpAggregator(K=1, method="bad")

    import deel.puncc.api.conformalization as conformalization_module

    class FakeCvPlusCalibrator:
        def __init__(self, calibrators):
            self.calibrators = calibrators

        def calibrate(self, X, kfold_predictors_dict, alpha):
            del kfold_predictors_dict, alpha
            X = np.asarray(X)
            return np.zeros(len(X)), np.ones(len(X))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(conformalization_module, "CvPlusCalibrator", FakeCvPlusCalibrator)
    try:
        cv_agg = CrossValCpAggregator(K=2)
        cv_agg._predictors = {0: dummy_predictor, 1: dummy_predictor}
        cv_agg._calibrators = {0: dummy_calibrator, 1: dummy_calibrator}
        y_pred, y_lo, y_hi = cv_agg.predict(np.array([[1.0], [2.0]]), alpha=0.1)
        assert y_pred is None
        np.testing.assert_allclose(y_lo, np.zeros(2))
        np.testing.assert_allclose(y_hi, np.ones(2))

        cv_agg.method = "bad"
        with pytest.raises(RuntimeError):
            cv_agg.predict(np.array([[1.0]]), alpha=0.1)
    finally:
        monkeypatch.undo()


def test_splitcp_and_enbpi_regression_branches(monkeypatch):
    cv_plus = CVPlus.__new__(CVPlus)
    with pytest.raises(RuntimeError):
        cv_plus.predict(np.array([[1.0]]), alpha=0.1)

    cv_plus2 = CVPlus.__new__(CVPlus)
    cv_plus2.conformal_predictor = DummyConformalPredictor()
    np.testing.assert_allclose(
        cv_plus2.get_nonconformity_scores()[0], np.array([1.0, 2.0])
    )

    enbpi = EnbPI(DummyPredictor(), B=1, random_state=7)
    assert enbpi.random_state == 7

    def resample_none(*args, **kwargs):
        del args, kwargs
        return None

    import deel.puncc.regression as regression_module

    monkeypatch.setattr(regression_module, "resample", resample_none)
    with pytest.raises(RuntimeError, match="Bootstrap dataset is empty"):
        enbpi.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    def resample_no_oob_for_first_unit(horizon_indices, **kwargs):
        del kwargs
        return np.zeros(len(horizon_indices), dtype=int)

    monkeypatch.setattr(regression_module, "resample", resample_no_oob_for_first_unit)
    with pytest.raises(RuntimeError, match='Increase "B"'):
        enbpi.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    enbpi = EnbPI(DummyPredictor(), B=1, agg_func_loo=lambda arr, axis=0: np.mean(arr, axis=axis))
    enbpi.residuals = [1.0, 2.0, 3.0]
    enbpi._boot_predictors = [DummyPredictor()]
    enbpi._oob_matrix = np.array([[1.0]])
    y_pred, y_lo, y_hi = enbpi.predict(
        np.array([[1.0], [2.0], [3.0], [4.0]]),
        alpha=0.1,
        y_true=np.array([1.0, 2.0, 3.0, 4.0]),
        s=2,
    )
    assert y_pred.shape == y_lo.shape == y_hi.shape == (4,)

    scalar_enbpi = EnbPI(DummyPredictor(), B=1, agg_func_loo=lambda arr, axis=0: np.array(1.5))
    scalar_enbpi.residuals = [1.0, 2.0]
    scalar_enbpi._boot_predictors = [DummyPredictor()]
    scalar_enbpi._oob_matrix = np.array([[1.0]])
    y_pred_scalar, _, _ = scalar_enbpi.predict(np.array([[1.0]]), alpha=0.1)
    assert y_pred_scalar.shape == (1,)
