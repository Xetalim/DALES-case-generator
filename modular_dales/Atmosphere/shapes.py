from dataclasses import dataclass, field
from typing import Self

import numpy as np

from modular_dales.modular.time_dependent_scalars import TimeDependentScalar
from modular_dales.MODULE_REGISTRY import register_module, register_special_serializing


@dataclass
class ShapeFunction:
    name: str

    # @staticmethod
    # def evaluate(*args, **kwargs):
    #     raise NotImplementedError("This method should be implemented by subclasses.")

    def params_at(self, time_value: float = 0.0) -> dict:
        return {
            key: (
                value.at(time_value)
                if isinstance(value, TimeDependentScalar)
                else value
            )
            for key, value in self.__dict__.items()
            if key != "name"
        }

    @property
    def params(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key != "name"}

    def at_time(self, time_value: float = 0.0) -> Self:
        params = self.params_at(time_value)
        return type(self)(**params)


@register_special_serializing
@register_module
@dataclass
class lin(ShapeFunction):
    """
    Linear shape function.

    Parameters
    ----------
    surf_val : float
        Surface value of the shape function.
    ddz : float
        Vertical gradient of the shape function.
    """

    name: str = field(default="lin", init=False)
    surf_val: float | TimeDependentScalar
    ddz: float | TimeDependentScalar

    def __init__(
        self, surf_val: float | TimeDependentScalar, ddz: float | TimeDependentScalar
    ):
        self.surf_val = surf_val
        self.ddz = ddz
        super().__init__(name="lin")

    @staticmethod
    def evaluate(z, surf_val, ddz):
        return surf_val + ddz * z


@register_special_serializing
@register_module
@dataclass
class expsinw(ShapeFunction):
    """
    Exponentially decaying sinusoidal shape function.

    Parameters
    ----------
    surf_val : float
        Surface value of the shape function.
    H : float
        Scale height for the exponential decay.
    amp : float
        Amplitude of the sinusoidal component.
    Hp : float
        Height above which the sinusoidal component is zero.
    """

    name: str = field(default="expsinw", init=False)
    surf_val: float | TimeDependentScalar
    H: float | TimeDependentScalar
    amp: float | TimeDependentScalar
    Hp: float | TimeDependentScalar

    def __init__(
        self,
        surf_val: float | TimeDependentScalar,
        H: float | TimeDependentScalar,
        amp: float | TimeDependentScalar,
        Hp: float | TimeDependentScalar,
    ):
        self.surf_val = surf_val
        self.H = H
        self.amp = amp
        self.Hp = Hp
        super().__init__(name="expsinw")

    @staticmethod
    def evaluate(z, surf_val, H, amp, Hp):
        wbase = -surf_val * (1 - np.exp(-z / H))
        wonion = amp * np.sin(2.0 * np.pi / Hp * z)
        wonion[z > Hp] = 0.0
        return wbase + wonion


@register_special_serializing
@register_module
@dataclass
class linmlsurf(ShapeFunction):
    """
    Linear mixed-layer surface shape function.

    Parameters
    ----------
    lapse_rate : float
        Lapse rate above the mixed-layer height.
    surf_val : float
        Surface value of the shape function.
    offset_val : float, optional
        Offset value below the mixed-layer height (default is 1.25).
    z_ml : float, optional
        Mixed-layer height (default is 500).
    """

    name: str = field(default="linmlsurf", init=False)
    lapse_rate: float
    surf_val: float | TimeDependentScalar
    offset_val: float | TimeDependentScalar = 1.25
    z_ml: float | TimeDependentScalar = 500

    def __init__(
        self,
        lapse_rate: float,
        surf_val: float | TimeDependentScalar,
        offset_val: float | TimeDependentScalar = 1.25,
        z_ml: float | TimeDependentScalar = 500,
    ):
        self.lapse_rate = lapse_rate
        self.surf_val = surf_val
        self.offset_val = offset_val
        self.z_ml = z_ml
        super().__init__(name="linmlsurf")

    @staticmethod
    def evaluate(z, lapse_rate, surf_val, offset_val=1.25, z_ml=500):
        u = np.zeros(z.shape)
        u[z <= z_ml] = surf_val - offset_val
        u[z > z_ml] = surf_val - offset_val + (z[z > z_ml] - z_ml) * lapse_rate
        return u


@register_special_serializing
@register_module
@dataclass
class exp(ShapeFunction):
    """
    Exponential shape function.

    Parameters
    ----------
    surf_val : float
        Surface value of the shape function.
    lambda_val : float
        Scale height for the exponential decay.
    z_ml : float, optional
        Reference height for the exponential decay (default is 500).
    """

    name: str = field(default="exp", init=False)
    surf_val: float | TimeDependentScalar
    lambda_val: float | TimeDependentScalar
    z_ml: float | TimeDependentScalar = 500

    def __init__(
        self,
        surf_val: float | TimeDependentScalar,
        lambda_val: float | TimeDependentScalar,
        z_ml: float | TimeDependentScalar = 500,
    ):
        self.surf_val = surf_val
        self.lambda_val = lambda_val
        self.z_ml = z_ml
        super().__init__(name="exp")

    @staticmethod
    def evaluate(z, surf_val, lambda_val, z_ml=500):
        u = surf_val * np.exp(-(z - z_ml) / lambda_val)
        u[z <= z_ml] = surf_val
        return u


ShapeSpec = lin | exp | expsinw | linmlsurf


SHAPE_FUNCTIONS = {
    "lin": lin.evaluate,
    "expsinw": expsinw.evaluate,
    "linmlsurf": linmlsurf.evaluate,
    "exp": exp.evaluate,
}
