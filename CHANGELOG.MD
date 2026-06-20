# Changelog

This file documents main changes across releases of **PUNCC**.

---

## [0.9.2] - 2026-06-20

### Added 
- Backend-level model copy abstraction supporting sklearn, JAX, TensorFlow/Keras, and PyTorch models through a unified `copy_model` entrypoint
- Added tutorial on how to use native tensorflow/keras models with puncc
- Added tutorial on how to use puncc with custom nonconformity scores and calibration workflows.
- Added `column_stack(...)` support across NumPy, pandas, PyTorch, JAX, and TensorFlow backend implementations
- Added `normalize_backend_inputs(...)` to infer a backend and normalize user-provided inputs once at API boundaries
- Added `empty_like(...)` support across NumPy, pandas, PyTorch, JAX, and TensorFlow backend implementations
- Added optional `like` support to backend `asarray(...)` methods to place auxiliary arrays on the same backend device as a reference object when supported
- Implemented `take(...)` for all registered backends to support splitting data across multiple data types.


### Changed
- Refactored predictor wrappers to delegate model cloning to the backend abstraction 
- Updated `TorchPredictor` experimental wrapper to reuse the shared backend copy mechanism
- Revised the PyTorch tutorial to use native torch models directly, removing the dependency on Skorch
- Updated CV conformalization to support GPU acceleration
- Avoid eager import (e.g. tensorflow) from prediction wrappers by detecting loaded frameworks through runtime module inspection
- Clarified the backend normalization contract in `backend.py` docstrings and helper utilities
- Updated `nonconformity_scores.py` and `prediction_sets.py` to reuse the shared backend normalization helper where backend inference semantics are preserved
- Updated the conformalization module so it no longer assumes NumPy arrays for inputs or outputs
- Updated `prediction_sets.py` to convert `scores_quantile` with `b.asarray(..., like=...)`, ensuring quantiles are placed on the same device as predictions for backend operations
- Updated `BasePredictor` and `DualPredictor` to support GPU acceleration   
- Updated `scaled_interval` to convert weights into the prediction backend before interval arithmetic
- Updated `CvPlusCalibrator` to convert nonconformity scores to NumPy before shape handling
- Updated weighted CV+ to append an inf block using the existing array rank instead of a 1-D scalar append
- Replaced the local splitter indexing fallback with backend delegation in `splitting` module, so splits now use backend-aware indexing instead of producing Python lists of row tensors
- Updated conformal prediction with tensorflow tutorial to use native TensorFlow data structures instead of numpy arrays


### Fixed
- Improved framework-agnostic model copy compatibility across predictor wrappers
- Corrected the definition of the prediction set for RAPS in theory overview 
- Completed backend `empty_like(...)` coverage and preserved pandas index/column structure for `empty_like` outputs
- Fixed mixed-device prediction set operations by aligning NumPy-derived `scores_quantile` values with the prediction tensor device for PyTorch, JAX, and TensorFlow backends
- Fixed `BasePredictor.predict` to use the backend abstraction for squeezing instead of calling `np.squeeze` directly
- Fixed TensorFlow asarray(..., like=...) in the backend so it aligns dtype with `like.dtype` in addition to device placement. This prevents for example float32/float64 interval arithmetic failures at prediction time.


### CI / Maintenance
- Expanded API test coverage for backend-driven model copying and predictor wrapper copy behavior.
- Added backend tests for `normalize_backend_inputs(...)` and `empty_like(...)`, plus regression 
coverage for updated nonconformity score and prediction set helpers
- Added backend tests covering `asarray(..., like=...)` behavior across NumPy, pandas, PyTorch, JAX, and TensorFlow, including JAX device compatibility paths and optional CUDA coverage for PyTorch
- Raised the minimum supported scikit-learn version to 1.5.0 to address security vulnerabilities in older releases.
- Added alternative expected seed results in conformal regression tests for newer scikit-learn versions.
- Removed py38 from tox envlist
- Added backend regression tests for torch/tensorflow same-device identity and torch device-move coverage.
- Added regression tests for torch weight handling, mixed torch/NumPy classwise LAC, and CV+ weight concatenation across folds.
- Added a backend regression test for row selection across backends
- Added a regression test for mixed NumPy/TensorFlow calibration inputs 
- Added TensorFlow, PyTorch and jax regression tests for dtype alignment in
---

## [0.9.1] - 2026-04-09

### Added
- LeverageWeightedCP (LWCP) for adaptive conformal regression using leverage-based weighting
- Classwise conformal classification (`ClasswiseLAC`) with associated metrics and tests
- Abstract backend layer for handling multiple data types
- New tutorials:
  - Adaptive conformal regression
  - Classwise classification
- Theoretical documentation for LWCP and Classwise LAC

### Changed
- Major refactor introducing `SplitConformalPredictor`
- Improved consistency across predictors, calibrators, and prediction sets
- Refactored internal backend for better extensibility

### Fixed
- Documentation issues and Sphinx build errors
- Visualization bugs and warnings

### CI / Maintenance
- Updated dependencies
- Improved CI/CD pipelines
- Added coverage badge

---

## [0.8.0] - 2024-11-05

### Added
- Improved compatibility with PyTorch-based workflows
- First structured tutorial suite covering core use cases
- Expanded examples for regression tasks

### Changed
- Refactored calibration pipeline for better modularity
- Improved API consistency across regression predictors
- Simplified usage patterns for common workflows

### Fixed
- Issues in prediction interval computation and edge cases in calibration
- Minor bugs in example notebooks

### Documentation
- Significant expansion of README and tutorials
- Clarified usage patterns for regression and adaptive methods

---

## [0.7.x] - 2024-06-15

### Added
- Conformal classification methods:
  - LAC (Least Ambiguous Classifier)
  - RAPS (Regularized Adaptive Prediction Sets)
- Metrics for evaluating prediction sets (coverage, size, etc.)
- Initial support for classification workflows

### Changed
- Refactoring of predictor and calibrator abstractions
- Improved modular separation between model, calibration, and prediction set construction
- Better handling of classification-specific score functions

### Fixed
- Stability issues in classification prediction sets
- Edge cases in multi-class handling

### Documentation
- Added classification examples and notebooks
- Improved explanation of prediction set evaluation

---

## [0.6.0] - 2024-03-20

### Added
- Time series conformal prediction utilities
- Dataset utilities and preprocessing helpers
- Early support for sequential data workflows

### Changed
- Improved internal organization of utilities and helpers
- Refactored data handling logic

### Fixed
- Minor inconsistencies in preprocessing pipelines

### Documentation
- Initial documentation for time series use cases
- Added examples demonstrating sequential prediction

---

## [0.5.0] - 2023-12-10

### Added
- First stable public API for regression conformal prediction
- Implementation of Split Conformal Prediction baseline
- Core calibration and scoring mechanisms
- Example notebooks for basic regression tasks

### Changed
- Stabilization of core interfaces (predictor, calibrator)
- Improved usability of regression workflows

### Fixed
- Early-stage bugs in calibration and interval construction

### Documentation
- Initial README structure
- Basic usage examples

---

## [0.4.0] - 2023-09-15

### Added
- Core architecture:
  - Predictor abstractions
  - Calibrator components
- Initial implementation of nonconformity scores
- Separation between model fitting and calibration

### Changed
- Iterative improvements of internal APIs
- Better separation of concerns across modules

### Notes
- This version marks the transition toward a more structured and extensible design

---

## [0.3.0] - 2023-06-10

### Added
- Implementation of conformal prediction methods
- Early prototypes for regression workflows
- Initial exploration of scoring strategies

---

## [0.2.0] - 2023-03-05

### Added
- Core utilities for conformal prediction
- Initial project structure and module organization

---

## [0.1.0] - 2023-01-20

### Added
- First public GitHub release of PUNCC following migration from the original GitLab repository
- Migration of an already existing conformal prediction codebase
- Core project structure, baseline implementations, and utilities carried over

### Notes
- PUNCC development started before this release in the original GitLab repository

---
