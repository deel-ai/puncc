import numpy as np

NUM_CLASSES = 1000

# L[y] is the per-class weight, initialized by calling init(seed)
L = None


def init(seed: int):
    """Draw the per-class weights L_y ~ Uniform(0,1). Must be called before
    using weighted_miscoverage_loss, with the same seed as the main experiment."""
    global L
    L = np.random.default_rng(seed).uniform(0.0, 1.0, size=NUM_CLASSES)


def weighted_miscoverage_loss(prediction_sets, labels) -> np.ndarray:
    """Per-sample loss  L_y * 1{y not in S}.

    Parameters
    ----------
    prediction_sets : list of array-like
        Each element is the predicted set S for one sample.
    labels : array-like of int
        Ground-truth class index y for each sample.

    Returns
    -------
    np.ndarray of shape (n,)
        Loss L_y if y is absent from S, 0 otherwise.
    """
    if L is None:
        raise RuntimeError("Call wmloss.init(seed) before using weighted_miscoverage_loss.")
    return np.array(
        [0.0 if y in S else float(L[y])
         for S, y in zip(prediction_sets, labels)]
    )
