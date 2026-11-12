"""Cliente mobile e inferência local do Apolo Zenith 1.9."""

from mobile.inference import ConfiguracaoMobile, MotorMobile, limitar_threads, quantizar_int8
from mobile.mobile_client import ClienteServidorApolo

__all__ = [
    "ConfiguracaoMobile",
    "MotorMobile",
    "limitar_threads",
    "quantizar_int8",
    "ClienteServidorApolo",
]
