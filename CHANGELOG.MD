# Changelog

This file documents main changes across releases of **PUNCC**.

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