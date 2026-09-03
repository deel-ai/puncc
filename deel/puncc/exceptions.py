class UncalibratedModelError(RuntimeError):
    default_message = """
        The model has not been calibrated, therefore, no prediction can be made.
        Please use model.calibrate(x_calib, y_calib).
    """
    def __init__(self, message:str|None = None):
        return super().__init__(message or self.default_message)