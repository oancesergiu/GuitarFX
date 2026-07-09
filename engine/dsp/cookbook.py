import math


def db_to_amplitude(db):
    return 10 ** (db / 40.0)


def peaking_eq(fc, fs, gain_db, q=1.0):
    A = db_to_amplitude(gain_db)

    omega = 2.0 * math.pi * fc / fs
    alpha = math.sin(omega) / (2.0 * q)
    cos_omega = math.cos(omega)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_omega
    b2 = 1.0 - alpha * A

    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_omega
    a2 = 1.0 - alpha / A

    return (
        b0 / a0,
        b1 / a0,
        b2 / a0,
        a1 / a0,
        a2 / a0,
    )


def low_shelf(fc, fs, gain_db, q=0.707):
    A = db_to_amplitude(gain_db)

    omega = 2.0 * math.pi * fc / fs
    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)

    alpha = sin_omega / (2.0 * q)
    sqrt_A = math.sqrt(A)

    b0 = A * ((A + 1) - (A - 1) * cos_omega + 2 * sqrt_A * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_omega)
    b2 = A * ((A + 1) - (A - 1) * cos_omega - 2 * sqrt_A * alpha)

    a0 = (A + 1) + (A - 1) * cos_omega + 2 * sqrt_A * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_omega)
    a2 = (A + 1) + (A - 1) * cos_omega - 2 * sqrt_A * alpha

    return (
        b0 / a0,
        b1 / a0,
        b2 / a0,
        a1 / a0,
        a2 / a0,
    )


def high_shelf(fc, fs, gain_db, q=0.707):
    A = db_to_amplitude(gain_db)

    omega = 2.0 * math.pi * fc / fs
    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)

    alpha = sin_omega / (2.0 * q)
    sqrt_A = math.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cos_omega + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_omega)
    b2 = A * ((A + 1) + (A - 1) * cos_omega - 2 * sqrt_A * alpha)

    a0 = (A + 1) - (A - 1) * cos_omega + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_omega)
    a2 = (A + 1) - (A - 1) * cos_omega - 2 * sqrt_A * alpha

    return (
        b0 / a0,
        b1 / a0,
        b2 / a0,
        a1 / a0,
        a2 / a0,
    )