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
"""
This module implements utility functions.
"""
import logging
import sys
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union
from scipy.optimize import linear_sum_assignment
import numpy as np

from deel.puncc.metrics import iou
from deel.puncc.api.backend import get_backend, shape2



logger = logging.getLogger(__name__)

EPSILON = sys.float_info.min  # small value to avoid underflow


def _n_samples(x: Any) -> int:
    """Return number of samples (first dimension) for array-like inputs."""
    try:
        return int(getattr(x, "shape")[0])
    except Exception:
        return len(x)


def _n_features(x: Any) -> int:
    """Return number of features (last dimension) for array-like inputs."""
    try:
        return int(getattr(x, "shape")[-1])
    except Exception:
        # last axis length for nested sequences
        if len(x) == 0:
            return 0
        return len(x[0])


def dual_predictor_check(l: Any, name: str, ltype: str):
    """Check if properties of `DualPredictor` come in
    couples.

    :param Any l: property of `DualPredictor`.
    :param str name: name of the property.
    :param str ltype: type of property.

    :raises TypeError: when the properties count is not two.

    """
    if (not isinstance(l, list)) or len(l) != 2:
        raise TypeError(
            f"Argument `{name}` should be a list of two {ltype}. Provided {l}."
        )


def logit_normalization_check(y: Iterable):
    """Check if provided logits sum is close to one.

    :param Iterable y: logits array. Rows correspond to samples and columns to
        classes.

    :raises ValueError: when logits sum is different than one within a tolerance
        value of 1e-5.
    """
    b = get_backend(y)
    y_arr = b.to_numpy(b.asarray(y))
    logits_sum = np.sum(y_arr, -1)
    if np.any(np.abs(logits_sum - 1) > 1e-5):
        raise ValueError(
            f"Logits must some to 1. Provided logit array {logits_sum}"
        )


def sample_len_check(a: Iterable, b: Iterable):
    """Check if arguments type have the same length.

    :param Iterable a: iterable whose type is supported.
    :param Iterable b: iterable whose type is supported.

    :raises TypeError: unsupported data types or elements have inconsistent types.
    :raises ValueError: arguments contain different number of samples.
    """
    supported_types_check(a, b)
    if _n_samples(a) != _n_samples(b):
        raise ValueError("Iterables must contain the same number of samples.")


def features_len_check(a: Iterable, b: Iterable):
    """Check if arguments have the same number of features,
    that is their last axes have the same length.

    :param Iterable a: iterable whose type is supported.
    :param Iterable b: iterable whose type is supported.

    :raises TypeError: unsupported data types or elements have inconsistent types.
    :raises ValueError: arguments have different number of features
    """

    supported_types_check(a, b)

    if _n_features(a) != _n_features(b):
        raise ValueError(
            "X_fit and X_calib must contain the same number of features."
        )


def supported_types_check(*data: Iterable):
    """Check if arguments' types are supported.

    Supported inputs are those handled by :func:`deel.puncc.api.backend.get_backend`,
    i.e. numpy arrays, pandas objects, torch/jax/tensorflow tensors (when already
    imported by the user), and Python sequences convertible to arrays.

    :param Iterable data: iterable(s) to be checked.

    :raises TypeError: unsupported data types.
    """
    for a in data:
        try:
            # Validation is delegated to backend inference + conversion.
            b = get_backend(a)
            _ = b.asarray(a)
        except Exception as e:
            raise TypeError(
                f"Unsupported data type {type(a)}. Please provide a numpy ndarray, "
                "a pandas object, a tensor, or an array-like convertible input."
            ) from e

def supported_meanvar_models_shape_check(Y_pred: Iterable, y_true: Iterable = None):
    """Check if arguments have the correct shape for mean-dispersion methods.
    Mean-dispersion methods require Y_pred to have shape (n_samples, 2) including
    two point predictions (e.g. mean and variance) and y_true to have shape
    (n_samples,).

    :param Iterable Y_pred: two point predictions for each sample.
    :param Iterable y_true: point observations for each sample.

    :raises TypeError: unsupported data types or elements have inconsistent types.
    :raises RuntimeError: Y_pred does not have shape (n_samples, 2)
        or y_true does not have shape (n_samples,).
    """
    supported_types_check(Y_pred, y_true)
    b = get_backend(Y_pred, y_true)
    Yp = b.asarray(Y_pred)
    yp_ndim, yp_shape = shape2(Yp)

    if yp_ndim != 2 or yp_shape[1] != 2:
        raise RuntimeError(
            "Each Y_pred must contain a point prediction and a "
            "dispersion estimation."
        )

    if y_true is None:
        return

    yt = b.asarray(y_true)
    yt_ndim, _ = shape2(yt)
    if yt_ndim != 1:
        raise RuntimeError("Each y_true must contain a point observation.")

def supported_dual_models_shape_check(Y_pred: Iterable, y_true: Iterable):
    """Check if arguments have the correct shape for dual-model methods.
    Dual-model methods require Y_pred to have shape (n_samples, 2) including
    two point predictions (e.g. mean and variance or lower and upper quantiles)
    and y_true to have shape (n_samples,).

    :param Iterable Y_pred: two point predictions for each sample.
    :param Iterable y_true: point observations for each sample.

    :raises TypeError: unsupported data types or elements have inconsistent types.
    :raises RuntimeError: Y_pred does not have shape (n_samples, 2)
        or y_true does not have shape (n_samples,).
    """
    supported_types_check(Y_pred, y_true)
    b = get_backend(Y_pred, y_true)
    Yp = b.to_numpy(Y_pred)
    yt = b.to_numpy(y_true)

    if Yp.ndim != 2 or Yp.shape[1] != 2:
        raise RuntimeError(
            "Each Y_pred must contain two point predictions "
            "(e.g. mean and variance or lower and upper quantiles)."
        )
    if yt.ndim != 1:
        raise RuntimeError("Each y_true must contain a point observation.")

def supported_bbox_shape_check(predicted_bboxes: Iterable, true_bboxes: Iterable):
    """Check if predicted and true bounding boxes have the same shape and the
    correct number of coordinates.

    :param Iterable predicted_bboxes: predicted bounding boxes.
    :param Iterable true_bboxes: true bounding boxes.

    :raises TypeError: unsupported data types.
    :raises RuntimeError: predicted and true bounding boxes have different shapes
        or do not contain 4 coordinates.
    """
    supported_types_check(predicted_bboxes, true_bboxes)

    b = get_backend(predicted_bboxes, true_bboxes)
    predicted_bboxes = b.asarray(predicted_bboxes)
    true_bboxes = b.asarray(true_bboxes)

    yp_ndim, yp_shape = shape2(predicted_bboxes)
    yt_ndim, yt_shape = shape2(true_bboxes)

    if yp_ndim != 2 or yp_shape[1] != 4:
        raise RuntimeError("Each predicted bounding box must contain "
                           "4 coordinates.")
    if yt_ndim != 2 or yt_shape[1] != 4:
        raise RuntimeError("Each true bounding box must contain 4 coordinates.")

def alpha_calib_check(
    alpha: Union[float, np.ndarray], n: int, complement_check: bool = False
):
    """Check if the value of alpha :math:`\\alpha` is consistent with the size
    of calibration set :math:`n`.

    The quantile order is inflated by a factor :math:`(1+1/n)` and has to be in
    the interval (0,1). From this, we derive the condition:

    .. math::

        0 < (1-\\alpha)\cdot(1+1/n) < 1 \implies 1 > \\alpha > 1/(n+1)

    If complement_check is set, we consider an additional condition:

    .. math::

        0 < \\alpha \cdot (1+1/n) < 1 \implies 0 < \\alpha < n/(n+1)

    :param float or np.ndarray alpha: target quantile order.
    :param int n: size of the calibration dataset.
    :param bool complement_check: complementary check to compute the
        :math:`\\alpha \cdot (1+1/n)`-th quantile, required by some methods
        such as jk+.

    :raises ValueError: when :math:`\\alpha` is inconsistent with the size of
        calibration set.
    """

    if np.any(alpha < 1 / (n + 1)):
        raise ValueError(
            f"Alpha is too small: (alpha={alpha}, n={n}) "
            + f"{alpha} < {1/(n+1)} for at least one of its coordinates. "
            + "Increase alpha or the size of the calibration set."
        )

    if np.any(alpha >= 1):
        raise ValueError(
            f"Alpha={alpha} is too large. "
            + "Decrease alpha such that all of its coordinates are < 1."
        )

    if complement_check and np.any(alpha > n / (n + 1)):
        raise ValueError(
            f"Alpha is too large: (alpha={alpha}, n={n}) "
            + f"{alpha} < {n/(n+1)} for at least on of its coordinates. "
            + "Decrease alpha or increase the size of the calibration set."
        )

    if complement_check and np.any(alpha <= 0):
        raise ValueError(
            f"Alpha={alpha} is too small. "
            + "Increase alpha such that all of its coordinates > 0."
        )


def get_min_max_alpha_calib(
    n: int, two_sided_conformalization: bool = False
) -> Optional[Tuple[float]]:
    """Get the greatest alpha in (0,1) that is consistent with the size of
    calibration set. Recall that while true Conformal Prediction
    (CP, Vovk et al 2005) is defined only on (0,1) (bounds not included), there
    maybe cases where we must handle alpha=0 or 1 (e.g. Adaptive Conformal
    Inference, Gibbs & Candès 2021).

    This function computes the admissible range of alpha for split CP
    (Papadopoulos et al 2002, Lei et al 2018) and the Jackknife-plus
    (Barber et al 2019). For more CP method, see the literature and update this
    function.

    In Split Conformal prediction we take the index
    := ceiling( (1-alpha)(n+1) )-th element from the __sorted__ nonconformity
    scores as the margin of error. We must have
    1 <= index <= n ( = len(calibration_set)) which boils down to:
    1/(n+1) <= alpha < 1.

    In the case of cv-plus and jackknife-plus (Barber et al 2019), we need:
        1. ceiling( (1-alpha)(n+1) )-th element of scores
        2. floor(     (alpha)(n+1) )-th element of scores

    The argument two_sided_conformalization=True, ensures the indexes are within
    range for this special case: 1/(n+1) <= alpha <= n/(n+1)

    :param int n: size of the calibration dataset.
    :param bool two_sided_conformalization: alpha threshold for two-sided
        quantile of jackknife+ (Barber et al 2021).

    :returns: lower and upper bounds for alpha (miscoverage probability)
    :rtype: Tuple[float]

    :raises ValueError: must have integer n, boolean two_sided_conformalization
        and n>=1
    """

    if n < 1:
        raise ValueError(f"Invalid input: you need n>=1 but received n={n}")

    if two_sided_conformalization is True:
        return (1 / (n + 1), n / (n + 1))

    # Base case: split conformal prediction (e.g. Lei et al 2019)
    return (1 / (n + 1), 1)


def quantile(
    a: Iterable,
    q: Union[float, np.ndarray],
    w: np.ndarray = None,
    axis: int = None,
    feature_axis: int = None,
) -> Union[float, np.ndarray]:
    """Estimate the columnwise q-th empirical weighted quantiles.

    :param Iterable a: collection of n samples
    :param Union[float, np.ndarray] q: q-th quantiles to compute. All elements must be in (0, 1).
    :param ndarray w: vector of size n. By default, w is None and equal weights
        (:math:`1/n`) are associated.
    :param int axis: axis along which to compute quantiles. If None,
        quantiles are computed along the flattened array.
    :param int feature_axis: if multidim quantile, feature_axis is the axis corresponding
        to the features.

    :raises ValueError: all coordinates of q must be in (0, 1).

    :returns: weighted empirical quantiles.
    :rtype: Union[float, np.ndarray]
    """
    # type checks:
    supported_types_check(a)

    b = get_backend(a)
    a = b.to_numpy(b.asarray(a))

    if w is not None:
        bw = get_backend(w)
        w = bw.to_numpy(bw.asarray(w))

# Sanity checks
    if np.any(q <= 0) or np.any(q >= 1):
        raise ValueError(
            "All coordinates of [q] must be in the open interval (0, 1)."
        )

    # Unweighted case
    if w is None:
        return quantile_unweighted(a, q, axis=axis, feature_axis=feature_axis)

    # Weighted case
    return quantile_weighted(a, q, w, axis=axis, feature_axis=feature_axis)


def quantile_unweighted(
    a: np.ndarray,
    q: Union[float, np.ndarray],
    axis: int = None,
    feature_axis: int = None,
) -> Union[float, np.ndarray]:
    """Estimate the multi-dimensional q-th empirical quantiles.

    :param ndarray a: collection of n samples
    :param Union[float, np.ndarray] q: q-th quantiles to compute. All elements must be in (0, 1).
    :param int axis: axis along which to compute quantiles. If None,
        quantiles are computed along all axes except the feature_axis.
    :param int feature_axis: if multidim quantile, feature_axis is the axis
        wich corresponds to the features.

    :raises ValueError: axis value cannot coincide with features axis.
    :raises ValueError: a and q must have the same number of features if q is an array.

    :returns: empirical quantiles.
    :rtype: Union[float, np.ndarray].
    """
    if axis is not None and axis == feature_axis:
        raise ValueError("axis value cannot coincide with features axis.")

    if isinstance(q, float):
        return np.quantile(a, q, axis=axis, method="inverted_cdf")

    if a.shape[feature_axis] != len(q):
        raise ValueError("a and q must have the same number of features.")
    quantile_res = np.array(
        [
            np.quantile(
                np.expand_dims(a.take(i, axis=feature_axis), axis=feature_axis),
                q[i],
                axis=axis,
                method="inverted_cdf",
            )
            for i in range(len(q))
        ]
    )
    return np.squeeze(
        np.transpose(quantile_res, (*range(1, quantile_res.ndim), 0))
    )


def quantile_weighted_unidim(
    a: np.ndarray, q: float, w: np.ndarray, axis=None
) -> Union[float, np.ndarray]:
    """Estimate the one-dimensional weighted q-th empirical quantiles.

    :param ndarray a: collection of n samples
    :param float: q-th quantiles to compute. All elements must be in (0, 1).
    :param ndarray w: array of weights.
    :param int axis: axis along which to compute quantiles. If None,
        quantiles are computed along the flattened array.

    :raises ValueError: w must be a 1D array.
    :raises ValueError: a and w must have the same length.

    :returns: empirical weighted quantiles.
    :rtype: Union[float, np.ndarray].
    """
    # Dimension checks
    if w.ndim != 1:
        raise ValueError("w must be a 1D array.")

    if len(w) != len(a):
        raise ValueError("a and w must have the same length.")

    # Normalization check
    norm_condition = np.isclose(np.sum(w) - 1, 0, atol=1e-14)
    if ~norm_condition:
        error = f"w is not normalized. Sum of weights is {np.sum(w)}"
        raise RuntimeError(error)

    # Values are sorted in ascending order
    sorted_idx = np.argsort(a, axis=axis)
    logger.debug("Sorted indices: %s", sorted_idx)

    # Reorder weights (ascending values of axis of a) and compute cumulative sum
    sorted_cumsum_weights = np.cumsum(w[sorted_idx], axis=axis)
    logger.debug("Sorted weights cumulative sum: %s", sorted_cumsum_weights)

    # Get the smallest index for which the cumulative sum of weights exceeds p
    min_idx_reaching_q = np.sum(
        sorted_cumsum_weights < q, axis=axis, keepdims=True
    )
    logger.debug(
        "First index per column where cumsum exceeds q: %s", min_idx_reaching_q
    )

    # Sort a
    sorted_a = np.take_along_axis(a, sorted_idx, axis=axis)

    # Collect the p-th quantile (first value whose probability mass exceeds p)
    quantile_res = np.take_along_axis(sorted_a, min_idx_reaching_q, axis=axis)
    logger.debug("Quantiles array: %s", quantile_res)
    return np.squeeze(quantile_res)


def quantile_weighted(
    a: np.ndarray,
    q: Union[float, np.ndarray],
    w: np.ndarray,
    axis: int = None,
    feature_axis: int = None,
) -> Union[float, np.ndarray]:
    """Estimate the multi-dimensional weighted q-th empirical quantiles.

    :param ndarray a: collection of n samples
    :param Union[float, np.ndarray] q: q-th quantiles to compute. All elements must be in (0, 1).
    :param ndarray w: array of weights.
    :param int axis: axis along which to compute quantiles. If None,
        quantiles are computed along the flattened array.
    :param int feature_axis: if multidim quantile, feature_axis is the axis
        wich corresponds to the features.

    :raises ValueError: cannot take quantiles along features axis.
    :raises ValueError: w must be have the same number of elements as a along axis.
    :raises ValueError: a and q must have the same number of features if q is an array.
    :raises ValueError: w and q must have the same number of features if w is a 2D array.

    :returns: empirical weighted quantiles.
    :rtype: Union[float, np.ndarray].
    """
    if axis is not None and axis == feature_axis:
        raise ValueError("axis value cannot coincide with features axis.")

    if axis is not None and len(w) != a.shape[axis]:
        raise ValueError("w must have the same length as a.shape[axis].")

    if isinstance(q, float) and w.ndim == 1:
        return quantile_weighted_unidim(a, q, w, axis=axis)

    if isinstance(q, float):
        quantile_res = np.array(
            [
                quantile_weighted_unidim(a[..., i], q, w[:, i], axis=axis)
                for i in range(a.shape[-1])
            ]
        )
    elif a.shape[-1] != len(q):
        raise ValueError("a and q must have the same number of features.")

    if w.ndim == 1:
        quantile_res = np.array(
            [
                quantile_weighted_unidim(a[..., i], q[i], w, axis=axis)
                for i in range(len(q))
            ]
        )
    elif w.ndim == 2 and w.shape[1] != len(q):
        raise ValueError("w and q must have the same number of features.")
    else:
        quantile_res = np.array(
            [
                quantile_weighted_unidim(
                    np.expand_dims(
                        a.take(i, axis=feature_axis), axis=feature_axis
                    ),
                    q[i],
                    w[..., i],
                    axis=axis,
                )
                for i in range(len(q))
            ]
        )

    return np.squeeze(
        np.transpose(quantile_res, (*range(1, quantile_res.ndim), 0))
    )


def hungarian_assignment(
    predicted_bboxes: Any, true_bboxes: Any, min_iou: float = 0.5
):
    """
    Assign predicted bounding boxes to labeled ones based on maximizing IOU.
    This function relies on the Hungarian algorithm (also known as the
    Kuhn-Munkres algorithm) to perform the assignment.

    :param predicted_bboxes: Array of predicted bounding boxes with
        shape (N, 4), where N is the number of predictions.
    :type predicted_bboxes: np.ndarray
    :param true_bboxes: Array of true bounding boxes with shape
        (M, 4), where M is the number of true classes.
    :type true_bboxes: np.ndarray
    :param min_iou: Minimum IoU threshold to consider a prediction as
        valid, by default 0.5.
    :type min_iou: float

    :return: A tuple containing 1) an array of aligned predicted bounding boxes
        that have IoU greater than the minimum threshold and 2) an array of
        true bounding boxes that correspond to the valid predicted bounding
        boxes.
    :rtype: tuple[np.ndarray, np.ndarray]

    .. note::
        This function pads the predicted bounding boxes to match the number of
        true bounding boxes if necessary. It then calculates the IoU matrix
        between true and predicted bounding boxes and performs linear sum
        assignment to maximize the total IoU. Finally, it filters out the
        bounding boxes that do not meet the minimum IoU threshold.

    Example
    -------
    .. code-block:: python

        >>> import numpy as np
        >>> predicted_bboxes = np.array([[10, 10, 50, 50], [20, 20, 60, 60]])
        >>> true_bboxes = np.array([[12, 12, 48, 48], [22, 22, 58, 58], [30, 30, 70, 70]])
        >>> hungarian_assignment(predicted_bboxes, true_bboxes, min_iou=0.5)
        (array([[10, 10, 50, 50], [20, 20, 60, 60]]), array([[12, 12, 48, 48], [22, 22, 58, 58]]))

    """

    b = get_backend(predicted_bboxes, true_bboxes)
    predicted_bboxes = b.to_numpy(b.asarray(predicted_bboxes))
    true_bboxes = b.to_numpy(b.asarray(true_bboxes))

    # Pad predicted bounding boxes to match the number of labeled ones
    def pad_predictions(predictions, labels):
        num_preds = predictions.shape[0]
        num_labels = labels.shape[0]

        if num_preds < num_labels:
            padded_predictions = np.zeros_like(labels)
            padded_predictions[:num_preds] = predictions
        else:
            padded_predictions = predictions.copy()

        return padded_predictions

    # Pad predicted bounding boxes to match the number of true bounding boxes
    padded_predictions = pad_predictions(predicted_bboxes, true_bboxes)

    # Calculate IoUs between true and predicted bounding boxes
    iou_matrix = np.round(iou(true_bboxes, padded_predictions), 2)

    # Perform linear sum assignment to maximize the total IoU
    _, best_pred_indices = linear_sum_assignment(iou_matrix, maximize=True)

    # Align predicted bounding boxes with true ones based on the best assignment
    aligned_predictions = padded_predictions[best_pred_indices]

    # Keep only those bounding boxes that have IoU greater than the minimum threshold
    valid_indices = iou(true_bboxes, aligned_predictions).diagonal() > min_iou

    return (
        aligned_predictions[valid_indices],
        true_bboxes[valid_indices],
        valid_indices,
    )

def generate_leverage_func(X):
    """
    Generate a leverage function based on the input data X. The leverage
    function is used to compute the leverage scores for provided samples.

    :param X: Train data from which to generate the leverage function.
        X has to be standardized and have more samples than features
        (n_samples > n_features) for the leverage scores to be well-defined.
    :type X: array-like

    :return: A function that takes as input a sample and returns its leverage score.
    :rtype: Callable
    """
    b = get_backend(X)
    X_np = b.to_numpy(b.asarray(X))
    # SVD decomposition of X
    _, S, Vt = np.linalg.svd(X_np, full_matrices=False)
    inv_S = np.linalg.inv(np.diag(S))
    leverage_matrix = inv_S @ Vt

    # The leverage function as defined from Fadnavis (2026)
    def leverage_function(x):
        b_x = get_backend(x)
        x_np = b_x.to_numpy(b_x.asarray(x))
        # compute norm L2 of leverage scores
        return np.linalg.norm(leverage_matrix @ x_np.T, ord=2, axis=0)

    return leverage_function
