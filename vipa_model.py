

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Default parameters from the MATLAB file
# ---------------------------------------------------------------------------

DEFAULTS = {
    "W": 25.4e-3,          # m, collimated beam diameter
    "fcyl": 0.200,         # m, cylindrical-camera focal length
    "n": 3.389,            # Si index at 4 K
    "h": 9.907e-3,         # m, VIPA thickness
    "theta_air_deg": 11.23, # deg, entrance angle
    "r1": np.sqrt(0.997),  # field reflectivity
    "L": 29.43e-3,         # m, VIPA length
    "fcam": 0.646,         # m
    "Spix": 0.035,         # mm, pixel size
    "WindowSize": 150e-3, # m
    "n_angles": 2500,
    "n_wavelengths": 101,
    "lambda_qcl_um": 79.701,
    "dLambda_dT_nm_K": 2.714,
}


# ---------------------------------------------------------------------------
# Load measured VIPA data
# ---------------------------------------------------------------------------

def load_vipa_data(data_dir):
    """
    Load the measured CSV data used by the MATLAB program.

    Returns
    -------
    vipa_data : ndarray
        Shape: (Nfiles, Npixels)

    qcl_temp_K : ndarray
        QCL temperatures extracted from filenames.

    files : list[Path]
        List of CSV files.
    """

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir.resolve()}"
        )

    rows = []
    qcl_temp = []

    for path in files:

        # MATLAB readmatrix equivalent for numeric CSV files
        data = np.genfromtxt(path, delimiter=",")

        if data.ndim == 1:
            data = data[np.newaxis, :]

        # MATLAB:
        # X = mean(vdata, 1);
        x = np.mean(data, axis=0)

        # MATLAB:
        # movmean(X, 5)
        #
        # Use a 5-point moving average.
        from scipy.ndimage import uniform_filter1d

        x_smooth = uniform_filter1d(
            x.astype(float),
            size=5,
            mode="nearest"
        )

        rows.append(x_smooth)

        # MATLAB:
        # tokens = regexp(files(p).name, '(\d+\.?\d*)K', 'tokens');
        match = re.search(
            r"(\d+\.?\d*)K",
            path.name
        )

        if match is None:
            raise ValueError(
                f"Could not extract QCL temperature from filename: "
                f"{path.name}\n"
                "Expected something like '79.7K'."
            )

        qcl_temp.append(
            float(match.group(1))
        )

    vipa_data = np.vstack(rows)
    qcl_temp = np.asarray(qcl_temp)

    return vipa_data, qcl_temp, files


# ---------------------------------------------------------------------------
# VIPA physics model
# ---------------------------------------------------------------------------

def calculate_model(
    n=DEFAULTS["n"],
    theta_air_deg=DEFAULTS["theta_air_deg"],
    lambda_qcl_um=DEFAULTS["lambda_qcl_um"],
    dLambda_dT_nm_K=DEFAULTS["dLambda_dT_nm_K"],
    DTemp=1.0,
    h=DEFAULTS["h"],
    L=DEFAULTS["L"],
    r1=DEFAULTS["r1"],
    fcam=DEFAULTS["fcam"],
    WindowSize=DEFAULTS["WindowSize"],
    n_angles=DEFAULTS["n_angles"],
    n_wavelengths=DEFAULTS["n_wavelengths"],
):
    """
    Calculate the VIPA power matrix and first two model peaks.

    This is the Python equivalent of the MATLAB physics section
    inside redraw().
    """

    theta_air = np.deg2rad(theta_air_deg)

    # -----------------------------------------------------------------------
    # Camera/output angular coordinate
    # -----------------------------------------------------------------------


    th_l_max = np.arctan(
        WindowSize / 2 / fcam
    )

    th_l = np.linspace(
        -th_l_max,
        th_l_max,
        n_angles
    )

    # -----------------------------------------------------------------------
    # Wavelength array
    # -----------------------------------------------------------------------


    lambda_um = np.linspace(
        lambda_qcl_um,
        dLambda_dT_nm_K * DTemp / 1000.0
        + lambda_qcl_um,
        n_wavelengths
    )

    wavelength_m = lambda_um * 1e-6

    # -----------------------------------------------------------------------
    # Physics
    # -----------------------------------------------------------------------


    r2 = np.sqrt(0.925)

    F1 = (
        4.0 * (r1 * r2)
        / (1.0 - (r1 * r2)) ** 2
    )


    kvec = (
        2.0 * np.pi / wavelength_m
    )[:, np.newaxis]


    ang_tot = (
        theta_air + th_l
    )

    sin_t = np.sin(
        ang_tot
    )[np.newaxis, :]

    sqrt_t = np.sqrt(
        n**2 - sin_t**2
    )

    theta_res = np.arcsin(
        sin_t / n
    )


    M = np.round(
        L
        / (
            2.0
            * h
            * np.tan(theta_res)
        )
    )

    # -----------------------------------------------------------------------
    # Phase
    # -----------------------------------------------------------------------

    delta = (
        2.0
        * h
        * (kvec * sqrt_t)
    )

    # -----------------------------------------------------------------------
    # Finesse / reflectivity term
    # -----------------------------------------------------------------------

    reflectivity = r1 * r2

    FM = (
        4.0
        * reflectivity**M
        / (
            1.0
            - reflectivity**M
        )**2
    )

    # -----------------------------------------------------------------------
    # Power
    # -----------------------------------------------------------------------


    Pwr = (
        1.0
        + FM
        * np.sin(
            M * delta / 2.0
        )**2
    ) / (
        1.0
        + F1
        * np.sin(
            delta / 2.0
        )**2
    )

    # -----------------------------------------------------------------------
    # Normalize each wavelength row
    # -----------------------------------------------------------------------


    norm_pwr = (
        Pwr
        / np.max(
            Pwr,
            axis=1,
            keepdims=True
        )
    )

    # -----------------------------------------------------------------------
    # Find peaks
    # -----------------------------------------------------------------------


    theta_pk_deg = np.full(
        (n_wavelengths, 2),
        np.nan
    )

    Pwr_pk = np.full(
        (n_wavelengths, 2),
        np.nan
    )

    for u in range(n_wavelengths):

        peaks, properties = find_peaks(
            norm_pwr[u, :],
            height=0.9
        )


        if len(peaks) >= 2:

            idx = peaks[:2]


            theta_pk_deg[u, :] = np.rad2deg(
                th_l[idx]
                + theta_air
            )

            Pwr_pk[u, :] = Pwr[
                u,
                idx
            ]

    return {
        "lambda_um": lambda_um,
        "wavelength_m": wavelength_m,
        "th_l_rad": th_l,
        "theta_res_rad": theta_res.ravel(),
        "M": M.ravel(),
        "delta": delta,
        "FM": FM.ravel(),
        "Pwr": Pwr,
        "norm_pwr": norm_pwr,
        "theta_pk_deg": theta_pk_deg,
        "Pwr_pk": Pwr_pk,
        "theta_air_deg": theta_air_deg,
    }


# ---------------------------------------------------------------------------
# Pixel -> angle conversion
# ---------------------------------------------------------------------------

def pixel_axis(
    n_pixels,
    Spix_mm=DEFAULTS["Spix"],
    fcam=DEFAULTS["fcam"]
):
    """
    Equivalent to the MATLAB:

        pixel_axis_ref =
            (1:size(vipa_data,2))
            - round(size(vipa_data,2)/2);

        ang = pixel_axis_ref
            * Spix*1e-3/fcam
            * 180/pi;
    """

    pixel_axis_ref = (
        np.arange(
            1,
            n_pixels + 1
        )
        - np.round(
            n_pixels / 2.0
        )
    )

    ang_deg = (
        pixel_axis_ref
        * Spix_mm
        * 1e-3
        / fcam
        * 180.0
        / np.pi
    )

    return pixel_axis_ref, ang_deg


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_model(
    n=DEFAULTS["n"],
    theta_air_deg=DEFAULTS["theta_air_deg"],
    lambda_qcl_um=DEFAULTS["lambda_qcl_um"],
    dLambda_dT_nm_K=DEFAULTS["dLambda_dT_nm_K"],
    DTemp=2.5
):
    """
    Convenience wrapper around calculate_model().
    """

    return calculate_model(
        n=n,
        theta_air_deg=theta_air_deg,
        lambda_qcl_um=lambda_qcl_um,
        dLambda_dT_nm_K=dLambda_dT_nm_K,
        DTemp=DTemp
    )


