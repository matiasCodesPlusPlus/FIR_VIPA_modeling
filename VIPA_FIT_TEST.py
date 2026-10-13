import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



# USER SETTINGS


IMAGE_DIR = Path(r".\\OUTPUT_MAPS")

# Set to a specific CSV filename to test only one image,
# or leave as None to process every CSV in IMAGE_DIR.
TEST_FILE = None

# Fit bounds and initial guess for r2
R2_INITIAL = np.sqrt(0.925)
R2_BOUNDS = (0.80, 0.9999)



# VIPA DEFAULTS -- KEEP PHYSICAL PARAMETERS FIXED


DEFAULTS = {
    "W": 25.4e-3,          # m
    "fcyl": 0.200,         # m
    "n": 3.389,            # Si index at 4 K
    "h": 9.907e-3,         # m, VIPA thickness
    "theta_air_deg": 11.23,
    "r1": np.sqrt(0.997),
    "L": 29.43e-3,         # m
    "fcam": 0.646,         # m
    "Spix": 0.035,         # mm per pixel
    "lambda_qcl_um": 79.701,  # wavelength at 50 K, um
    "dLambda_dT_nm_K": 2.714,  # nm/K
}

THETA_AIR = np.deg2rad(DEFAULTS["theta_air_deg"])


# WAVELENGTH CALIBRATION


def wavelength_from_temperature(temperature_K):
    """
    Known QCL wavelength calibration.
    Reference: 79.701 um at 50 K.
    dLambda/dT = 2.714 nm/K.
    Returns wavelength in meters.
    """
    wavelength_um = (
        DEFAULTS["lambda_qcl_um"]
        + DEFAULTS["dLambda_dT_nm_K"]
        * (temperature_K - 50.0) / 1000.0
    )
    return wavelength_um * 1e-6



# VIPA PHYSICS MODEL -- UNCHANGED


def vipa_power(theta_l, wavelength_m, r1, r2, theta_air):
    angle_tot = theta_air + theta_l
    sin_t = np.sin(angle_tot)
    sqrt_t = np.sqrt(DEFAULTS["n"]**2 - sin_t**2)
    theta_res = np.arcsin(sin_t / DEFAULTS["n"])

    M = np.round(
        DEFAULTS["L"]
        / (2.0 * DEFAULTS["h"] * np.tan(theta_res))
    )

    k = 2.0 * np.pi / wavelength_m
    delta = 2.0 * DEFAULTS["h"] * k * sqrt_t
    R = r1 * r2

    F1 = 4.0 * R / (1.0 - R)**2
    FM = 4.0 * R**M / (1.0 - R**M)**2

    return (
        1.0 + FM * np.sin(M * delta / 2.0)**2
    ) / (
        1.0 + F1 * np.sin(delta / 2.0)**2
    )



# PIXEL-TO-ANGLE CONVERSION


def pixel_angles(n_pixels):
    """
    Pixel 0 center is x = -1.72 mm.
    Pixel spacing is 0.035 mm.
    Zero-angle reference is x = 50 mm.
    """
    x_mm = -1.72 + np.arange(n_pixels) * DEFAULTS["Spix"]
    x_rel_m = (x_mm - 50.0) * 1e-3

    return np.arctan(x_rel_m / DEFAULTS["fcam"]), x_mm



# FITTING



def fit_profile(image, temperature_K):
    # Vertically collapse the image.
    profile = np.sum(image, axis=0).astype(float)

    theta_l, x_mm = pixel_angles(profile.size)
    wavelength_m = wavelength_from_temperature(temperature_K)

    valid = np.isfinite(profile) & np.isfinite(theta_l)
    theta_fit = theta_l[valid]
    y_fit = profile[valid]
    # Normalize measured profile to its actual maximum
    data_max = np.max(y_fit)
    y_norm = y_fit / data_max


    #---------------------------------------------------------------------
    def normalized_vipa_power(theta, r2, theta_air):
        power = vipa_power(
            theta,
            wavelength_m,
            DEFAULTS["r1"],
            r2,
            theta_air,
        )

        pmax = np.max(power)
        if not np.isfinite(pmax) or pmax <= 0:
            raise ValueError("Invalid VIPA power normalization.")

        return power / pmax


    def fit_model(theta, r2, theta_air, offset):
        return normalized_vipa_power(theta, r2, theta_air) + offset


    p0 = [R2_INITIAL, THETA_AIR, 0.0]

    bounds = (
        [R2_BOUNDS[0], np.deg2rad(9.0), -np.inf],
        [R2_BOUNDS[1], np.deg2rad(13.0), np.inf],
    )

    params, covariance = curve_fit(
        fit_model,
        theta_fit,
        y_norm,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )

    r2_fit, theta_air_fit, offset_norm = params

    fitted_norm = fit_model(theta_fit, *params)

    fitted_profile = np.full(profile.shape, np.nan)
    fitted_profile[valid] = fitted_norm * data_max

    residuals = y_fit - fitted_profile[valid]
    rmse = np.sqrt(np.mean(residuals**2))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"Fitted entrance angle: {np.rad2deg(theta_air_fit):.5f} deg")
    return {
    "x_mm": x_mm,
    "profile": profile,
    "fit": fitted_profile,
    "r2": r2_fit,
    "amplitude": data_max,
    "offset": offset_norm * data_max,
    "covariance": covariance,
    "rmse": rmse,
    "r_squared": r_squared,
    "wavelength_um": wavelength_m * 1e6,
    "temperature_K": temperature_K,
    "theta_air_deg_fit": np.rad2deg(theta_air_fit),
}


# LOAD AND PROCESS IMAGES


if TEST_FILE is not None:
    image_paths = [IMAGE_DIR / TEST_FILE]
else:
    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    )

if not image_paths:
    raise FileNotFoundError(f"No CSV images found in {IMAGE_DIR}")

for image_path in image_paths:
    # Expected filename ending: ..._50.0.csv, ..._52.5.csv, etc.
    temperature_K = float(
        image_path.name.split("_")[-1].strip(".csv")
    )

    print(f"\nProcessing: {image_path.name}")
    print(f"Temperature: {temperature_K:.2f} K")

    image = np.fliplr(np.loadtxt(image_path, delimiter=","))

    try:
        result = fit_profile(image, temperature_K)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        print(f"Fit failed: {exc}")
        continue

    print(f"Wavelength: {result['wavelength_um']:.6f} um")
    print(f"Fitted r2:  {result['r2']:.7f}")
    print(f"Amplitude:  {result['amplitude']:.6g}")
    print(f"Offset:     {result['offset']:.6g}")
    print(f"R-squared:  {result['r_squared']:.6f}")
    print(f"RMSE:       {result['rmse']:.6g}")
    
    
    # Plot fit and full image
    
    fig, (ax_profile, ax_image) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )

    ax_profile.plot(
        result["x_mm"],
        result["profile"],
        label="Measured vertical sum",
        linewidth=1.2,
    )
    ax_profile.plot(
        result["x_mm"],
        result["fit"],
        "--",
        label="VIPA model fit",
        linewidth=1.5,
    )

    ax_profile.set_xlabel("Output-plane coordinate (mm)")
    ax_profile.set_ylabel("Vertically summed signal (a.u.)")
    ax_profile.set_title(
        f"{image_path.name} | T={temperature_K:.2f} K | "
        f"R²={result['r_squared']:.4f} | "
        f"RMSE={result['rmse']:.4g}"
    )
    ax_profile.grid(True, alpha=0.3)
    ax_profile.legend()

    ax_image.imshow(
        image,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )
    ax_image.set_xlabel("Pixel column")
    ax_image.set_ylabel("Pixel row")
    ax_image.set_title("Original image")

    plt.show()
