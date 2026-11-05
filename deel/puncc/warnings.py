class PunccWarning(UserWarning):
    """Base warning for PUNCC."""


class CalibrationWarning(PunccWarning):
    """Warning related to calibration or statistical guarantees."""


class NumericalWarning(PunccWarning):
    """Warning related to numerical stability."""


class BackendWarning(PunccWarning):
    """Warning related to backend-specific behavior."""


class ExperimentalWarning(PunccWarning):
    """Warning for experimental or partially supported features."""


class ModelCloningWarning(PunccWarning):
    """Warning related to potentially imperfect model cloning."""
