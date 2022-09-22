Predictive UNcertainty Calibration and Conformalization (PUNCC) is an open-source library that enables ad-hoc integration of AI models into a theoretically sound uncertainty estimation framework based on conformal prediction. Prediction sets are constructed with guaranteed coverage probability according to a nominal level of error $\alpha$.

# Installation

## Clone the repo

```bash
git clone ssh://git@forge.deel.ai:22012/statistic-guarantees/puncc.git
```

## Installation

It is recommended to install puncc in a virtual environment to not mess with you system dependencies.

### Using pip

#### For developpers

```bash
pip install -e .[dev]
```

#### For users
```bash
pip install -e .[interactive]
```

### Using makefile

```bash
make prepare-dev
```

### From Makefile

```bash
make prepare-dev
```

# Quickstart

We propose two ways of defining and using conformal prediction wrappers:
- A fast way based on preconfigured conformal prediction wrappers
- A flexible way based of full customization of the prediction model, the residual computation and the fit/calibration data split plan

A comparison of both approaches is provided [here](docs/quickstart.ipynb) for a simple regression problem. A showcase of basic uncertainty quantification features for the first approach is given hereafter.

## Data
```python
import numpy as np
from sklearn import datasets

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
X_train = diabetes_X[:-100]
X_test = diabetes_X[-100:]

# Split the targets into training/testing sets
y_train = diabetes_y[:-100]
y_test = diabetes_y[-100:]

# Split fit and calibration data
X_fit, X_calib = X_train[:-100], X_train[-100:]
y_fit, y_calib = y_train[:-100], y_train[-100:]
```
## Linear regression model

```python
from sklearn import linear_model

# Create linear regression model
regr = linear_model.LinearRegression()
```


## Split conformal prediction
``` python
from deel.puncc.regression import SplitCP

# Coverage target is 1-alpha = 90%
alpha=.1
# Instanciate the split cp wrapper on the linear model
split_cp = SplitCP(regr)
# Train model on the fitting dataset and compute residuals on the calibration
# dataset
split_cp.fit(X_fit, y_fit, X_calib, y_calib)
# Estimate the prediction interval
y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)
```

## Results

```python
from deel.puncc.api.utils import plot_prediction_interval

# Figure of the prediction bands

plot_prediction_interval(
    X = X_test[:,0],
    y_true=y_test,
    y_pred=y_pred,
    y_pred_lower=y_pred_lower,
    y_pred_upper=y_pred_upper,
    sort_X=True,
    size=(10, 6),
    loc="upper left")
```

![90% Prediction Interval with the Split Conformal Prediction Method](docs/source/results_quickstart_split_cp_pi.png)

## Citation

This library was built to support the work presented in our COPA 2022 paper on conformal prediction for timeseries. If you use our library for your work, please cite our paper:

```
@inproceedings{mendil2022robust,
  title={Robust Gas Demand Forecasting With Conformal Prediction},
  author={Mendil, Mouhcine and Mossina, Luca and Nabhan, Marc and Pasini, Kevin},
  booktitle={Conformal and Probabilistic Prediction with Applications},
  pages={169--187},
  year={2022},
  organization={PMLR}
}
```

## Contributing

Contributions are welcome! Feel free to report an issue or open a pull
request. Take a look at our guidelines [here](CONTRIBUTING.md).

## License

Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry, CRIAQ
and ANITI - https://www.deel.ai/

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgments

This project received funding from the French "Investing for the Future – PIA3" program
within the Artiﬁcial and Natural Intelligence Toulouse Institute (ANITI). The authors
gratefully acknowledge the support of the [DEEL project](https://www.deel.ai/).
