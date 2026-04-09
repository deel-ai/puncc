import torch
import torch.nn.functional as F
import numpy as np


class LambdaPredictor:
    """Family of set-valued classifiers T_lambda(x) = {y : pi_x(y) >= -lambda}.

    Wraps any PyTorch softmax classifier into a LambdaPredictor compatible
    with RCPS. For each threshold lambda in [0, 1]:
      - lambda = -1.0  ->  empty sets
      - lambda = 0.0  ->  all classes included (maximum sets)

    X can be either:
      - a torch.Tensor of shape (n, C, H, W)  [raw images, runs the model]
      - a numpy.ndarray of shape (n, num_classes)  [pre-computed softmax scores]

    Parameters
    ----------
    backbone : torch.nn.Module
        A pretrained classifier whose forward pass returns class logits.
    device : str
        Device to run inference on ('cuda' or 'cpu').
    """

    def __init__(self, backbone: torch.nn.Module, device: str = "cuda"):
        self.backbone = backbone
        self.device = device

    def _softmax_scores(self, X) -> np.ndarray:
        if isinstance(X, np.ndarray):
            # Already pre-computed softmax scores
            return X
        # Raw image tensors: run the model
        self.backbone.eval()
        with torch.no_grad():
            logits = self.backbone(X.to(self.device))
            scores = F.softmax(logits, dim=1).cpu().numpy()
        return scores

    def predict(self, X, lam: float):
        """Return T_lambda(x) = {y : pi_x(y) >= -lambda} for each sample in X."""
        scores = self._softmax_scores(X)
        return [np.where(row >= -lam)[0] for row in scores]

    def __call__(self, X, lam: float):
        return self.predict(X, lam)


class LambdaPredictorFromSoftmax:
    """LambdaPredictor wrapper for pre-computed softmax scores.

    This is a thin wrapper around LambdaPredictor that allows using pre-computed
    softmax scores directly, without needing to run a model. It assumes that the
    input X is already a numpy array of shape (n, num_classes) containing the
    softmax scores for each sample.

    Parameters
    ----------
    None
    """

    def __init__(self):
        pass

    def predict(self, X, lam: float):
        """Return T_lambda(x) = {y : pi_x(y) >= -lambda} for each sample in X."""
        return [np.where(row >= -lam)[0] for row in X]

    def __call__(self, X, lam: float):
        return self.predict(X, lam)
    


class LambdaPredictorSetSizeFromSofmax:
    """Predict set sizes |T_lambda(x)| = number of classes with pi_x(y) >= -lambda.

    This is a thin wrapper around LambdaPredictor that returns the size of the
    predicted sets instead of the sets themselves. It assumes that the input X
    is already a numpy array of shape (n, num_classes) containing the softmax
    scores for each sample.

    Parameters
    ----------
    None
    """

    def __init__(self):
        pass

    def predict(self, X, lam: float):
        """Return |T_lambda(x)| = number of classes with pi_x(y) >= -lambda."""
        return np.sum(X >= -lam, axis=1)

    def __call__(self, X, lam: float):
        return self.predict(X, lam)
