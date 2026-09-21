"""Parameter validation shared by plans and models."""
from numbers import Integral
def integer(name, value, lower):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
