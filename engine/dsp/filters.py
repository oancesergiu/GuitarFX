import math


def lowpass_coefficients(fc, fs, q=0.707):
    omega = 2.0 * math.pi * fc / fs

    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)

    alpha = sin_omega / (2.0 * q)

    b0 = (1 - cos_omega) / 2
    b1 = 1 - cos_omega
    b2 = (1 - cos_omega) / 2

    a0 = 1 + alpha
    a1 = -2 * cos_omega
    a2 = 1 - alpha

    return (
        b0 / a0,
        b1 / a0,
        b2 / a0,
        a1 / a0,
        a2 / a0,
    )