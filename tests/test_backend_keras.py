from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys

import pytest


def test_tensorflow_uniform_weighted_quantile_accepts_scalar():
    if find_spec("tensorflow") is None:
        pytest.skip("TensorFlow is not installed.")

    repo_root = Path(__file__).resolve().parents[1]
    script = """
from deel.puncc.backend import ops
from deel.puncc.config import set_backend

set_backend("tensorflow")
values = ops.convert_to_tensor([[1.0], [2.0], [3.0], [4.0]])
result = ops.weighted_quantile(values, 0.5, axis=0)

assert tuple(result.shape) == (1,)
assert ops.tolist(result) == [2.0]
"""

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=repo_root,
    )
