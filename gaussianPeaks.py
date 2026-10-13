import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks
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


# USER SETTINGS---------------------------------

X_MIN_MM = 5.0
X_MAX_MM = 85.0

# Center of focal-plane coordinate system
X_CENTER_MM = 50.0

# Pixel pitch
PIXEL_PITCH_MM = 0.035

# Additional pixels included on either side
EXTRA_PIXELS = 192

# Peak detection
PROMINENCE_FRACTION = 0.01
MIN_PEAK_DISTANCE_MM = 3.0

# Gaussian width limits
MIN_SIGMA_MM = 0.05
MAX_SIGMA_MM = 5.0

# Approximate initial Gaussian width
INITIAL_SIGMA_MM = 0.5

# Size of fitting window around candidate peaks
FIT_WINDOW_MM = 5.0

# Require this much separation between two fitted peaks
MIN_PEAK_SEPARATION_MM = 1.5

# If True, show every fit
SHOW_EACH_FIT = True


# VIPA ORDER SPACING----------------------------------------------

EXPECTED_ORDER_SPACING_MM = 48.0

ORDER_SEARCH_HALF_WIDTH_MM = 4.0

# Minimum amplitude for the second order relative to the
# first order.
MIN_SECOND_ORDER_FRACTION = 0.01


VIPA = {
    "n": 3.389,
    "h": 9.907e-3,          # m
    "L": 29.43e-3,          # m
    "fcam": 0.646,          # m
    "theta_air_deg": 11.23,
    "lambda_50K_um": 79.701,
    "dLambda_dT_nm_K": 2.714,
}

R1_FIXED = np.sqrt(0.997)

#T to LAMBDA
def wavelength_from_temperature(T):
    return (
        VIPA["lambda_50K_um"]
        + VIPA["dLambda_dT_nm_K"] * (T - 50.0) / 1000.0
    ) * 1e-6

#SINGLE GAUSSIAN FUNC
def gaussian(x, y0, A, x0, sigma):
    """
    Single Gaussian + constant background.
    """
    return (
        y0
        + A * np.exp(
            -((x - x0) ** 2)
            / (2.0 * sigma ** 2)
        )
    )

#DOUBLE GAUSSIAN FUNC
def double_gaussian(
    x,
    y0,
    A1,
    x01,
    sigma1,
    A2,
    x02,
    sigma2,
):
    """
    Two Gaussians + constant background.
    """
    return (
        y0
        + A1 * np.exp(
            -((x - x01) ** 2)
            / (2.0 * sigma1 ** 2)
        )
        + A2 * np.exp(
            -((x - x02) ** 2)
            / (2.0 * sigma2 ** 2)
        )
    )

# FIND CANDIDATE PEAKS
def find_candidate_peaks(
    x,
    y,
    prominence_fraction=PROMINENCE_FRACTION,
    min_distance_mm=MIN_PEAK_DISTANCE_MM,
):
    """
    Find candidate peaks.

    This is intentionally permissive. The purpose is to
    generate possible peak locations, NOT decide whether
    the peaks are real VIPA orders.
    """

    y_range = np.max(y) - np.min(y)

    prominence = (
        prominence_fraction * y_range
    )

    dx = np.mean(np.diff(x))

    min_distance_pixels = max(
        1,
        int(min_distance_mm / dx)
    )

    indices, properties = find_peaks(
        y,
        prominence=prominence,
        distance=min_distance_pixels,
    )

    return indices, properties

# FIT SINGLE GAUSSIAN
def fit_single_gaussian(
    x,
    y,
    peak_index,
):
    """
    Fit one Gaussian around a candidate peak.
    """

    x_peak = x[peak_index]

    mask = (
        np.abs(x - x_peak)
        <= FIT_WINDOW_MM / 2.0
    )

    x_fit = x[mask]
    y_fit = y[mask]

    if len(x_fit) < 5:
        raise RuntimeError(
            "Not enough points for Gaussian fit."
        )

    # Baseline estimate
    y0_guess = np.percentile(
        y_fit,
        10
    )

    A_guess = max(
        y[peak_index] - y0_guess,
        1e-12
    )

    p0 = [
        y0_guess,
        A_guess,
        x_peak,
        INITIAL_SIGMA_MM,
    ]

    bounds = (
        [
            -np.inf,
            0.0,
            x_fit.min(),
            MIN_SIGMA_MM,
        ],
        [
            np.inf,
            np.inf,
            x_fit.max(),
            MAX_SIGMA_MM,
        ],
    )

    popt, pcov = curve_fit(
        gaussian,
        x_fit,
        y_fit,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )

    y_model = gaussian(
        x_fit,
        *popt
    )

    rss = np.sum(
        (y_fit - y_model) ** 2
    )

    

    return {
        "popt": popt,
        "pcov": pcov,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "y_model": y_model,
        "rss": rss,
    }

# FIT DOUBLE GAUSSIAN
def fit_double_gaussian(x, y, x1, x2):

    y0 = np.percentile(y, 10)

    A1 = y[np.argmin(abs(x - x1))] - y0
    A2 = y[np.argmin(abs(x - x2))] - y0

    p0 = [
        y0,
        max(A1, 1e-12),
        x1,
        INITIAL_SIGMA_MM,
        max(A2, 1e-12),
        x2,
        INITIAL_SIGMA_MM,
    ]

    bounds = (
        [
            -np.inf,
            0,
            x1 - 5,
            MIN_SIGMA_MM,
            0,
            x2 - ORDER_SEARCH_HALF_WIDTH_MM,
            MIN_SIGMA_MM,
        ],
        [
            np.inf,
            np.inf,
            x1 + 5,
            MAX_SIGMA_MM,
            np.inf,
            x2 + ORDER_SEARCH_HALF_WIDTH_MM,
            MAX_SIGMA_MM,
        ],
    )

    popt, pcov = curve_fit(
        double_gaussian,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )

    return popt, pcov

#MULTI GAUSSIAN FIT
def fit_all_profiles(
    profiles,
    fullmaps,
    temperatures,
    show_plots=SHOW_EACH_FIT,
):
    """
    Fit every flattened focal-plane image.

    Parameters
    ----------
    profiles : ndarray
        Shape (N_images, N_pixels)

    temperatures : ndarray
        Shape (N_images,)

    Returns
    -------
    results : list of dictionaries
    """

    profiles = np.asarray(
        profiles,
        dtype=float
    )

    temperatures = np.asarray(
        temperatures,
        dtype=float
    )

    if profiles.ndim != 2:
        raise ValueError(
            "profiles must be a 2-D array: "
            "(N_images, N_pixels)"
        )

    if len(temperatures) != len(profiles):
        raise ValueError(
            "Number of temperatures must "
            "match number of profiles."
        )

    n_images, n_pixels = (
        profiles.shape
    )

    # --------------------------------------------------------
    # Construct focal-plane coordinate
    # --------------------------------------------------------

    # Position of the first actual pixel
    x_start = (
        X_MIN_MM
        - EXTRA_PIXELS * PIXEL_PITCH_MM
    )

    x = (
        x_start
        + np.arange(n_pixels)
        * PIXEL_PITCH_MM
    )

    # Shift center to 0 mm
    x_relative = (
        x - X_CENTER_MM
    )

    # --------------------------------------------------------
    # Fit each profile
    # --------------------------------------------------------

    results = []

    for i in range(n_images):

        print(
            f"Fitting "
            f"{i + 1}/{n_images}..."
        )

        result = fit_vipa_profile(
            x_relative,
            profiles[i],
        )

        result["index"] = i
        result["temperature_K"] = (
            temperatures[i]
        )

        results.append(
            result
        )

        if show_plots:

            plot_gaussian_fit(
                x_relative,
                profiles[i],
                fullmaps[i],
                result,
                temperature=temperatures[i],
            )

    return (
        x_relative,
        results
    )

#FIT VIPA PROFILE
def fit_vipa_profile(x, y):

    # --------------------------------------------------------
    # Find strongest peak
    # --------------------------------------------------------

    peaks, _ = find_candidate_peaks(x, y)

    if len(peaks) == 0:
        return {
            "success": False,
            "n_peaks": 0,
        }

    i1 = peaks[np.argmax(y[peaks])]

    # --------------------------------------------------------
    # Fit first Gaussian
    # --------------------------------------------------------

    fit1 = fit_single_gaussian(
        x,
        y,
        i1,
    )

    p1 = fit1["popt"]

    x01 = p1[2]

    # --------------------------------------------------------
    # Expected position of second order
    # --------------------------------------------------------

    x02_guess = (
        x01
        - EXPECTED_ORDER_SPACING_MM
    )

    # --------------------------------------------------------
    # Look for signal around expected second-order position
    # --------------------------------------------------------

    mask = (
        np.abs(x - x02_guess)
        <= ORDER_SEARCH_HALF_WIDTH_MM
    )

    if not np.any(mask):

        return {
            "success": True,
            "n_peaks": 1,
            "y0": p1[0],
            "A1": p1[1],
            "x01": p1[2],
            "sigma1": p1[3],
            "A2": np.nan,
            "x02": np.nan,
            "sigma2": np.nan,
            "fit": fit1,
            "fit_type": "single",
        }

    # Position of maximum signal in expected region
    indices = np.where(mask)[0]

    i2 = indices[
        np.argmax(y[indices])
    ]

    x02 = x[i2]

    # --------------------------------------------------------
    # Is there actually a meaningful second order?
    # --------------------------------------------------------

    second_amplitude = (
        y[i2] - p1[0]
    )

    if second_amplitude <= (
        MIN_SECOND_ORDER_FRACTION
        * p1[1]
    ):

        return {
            "success": True,
            "n_peaks": 1,
            "y0": p1[0],
            "A1": p1[1],
            "x01": p1[2],
            "sigma1": p1[3],
            "A2": np.nan,
            "x02": np.nan,
            "sigma2": np.nan,
            "fit": fit1,
            "fit_type": "single",
        }

    # --------------------------------------------------------
    # Fit both orders
    # --------------------------------------------------------

    try:

        p2, pcov2 = fit_double_gaussian(
            x,
            y,
            x01,
            x02,
        )

    except RuntimeError:

        return {
            "success": True,
            "n_peaks": 1,
            "y0": p1[0],
            "A1": p1[1],
            "x01": p1[2],
            "sigma1": p1[3],
            "A2": np.nan,
            "x02": np.nan,
            "sigma2": np.nan,
            "fit": fit1,
            "fit_type": "single",
        }

    # --------------------------------------------------------
    # Return two-order fit
    # --------------------------------------------------------

    return {
        "success": True,
        "n_peaks": 2,

        "y0": p2[0],

        "A1": p2[1],
        "x01": p2[2],
        "sigma1": p2[3],

        "A2": p2[4],
        "x02": p2[5],
        "sigma2": p2[6],

        "fit": {
            "popt": p2,
            "pcov": pcov2,
        },

        "fit_type": "double",
    }

# CONVERT RESULTS TO ARRAYS
def results_to_arrays(results):
    """
    Convert fitting results into convenient NumPy arrays.
    """

    temperature = np.array([
        r["temperature_K"]
        for r in results
    ])

    n_peaks = np.array([
        r["n_peaks"]
        for r in results
    ])

    x01 = np.array([
        r["x01"]
        for r in results
    ])

    A1 = np.array([
        r["A1"]
        for r in results
    ])

    sigma1 = np.array([
        r["sigma1"]
        for r in results
    ])

    x02 = np.array([
        r["x02"]
        for r in results
    ])

    A2 = np.array([
        r["A2"]
        for r in results
    ])

    sigma2 = np.array([
        r["sigma2"]
        for r in results
    ])

    return {
        "temperature_K": temperature,
        "n_peaks": n_peaks,

        "x01_mm": x01,
        "A1": A1,
        "sigma1_mm": sigma1,

        "x02_mm": x02,
        "A2": A2,
        "sigma2_mm": sigma2,
    }

# PLOT GAUSSIAN FITS
def plot_gaussian_fit(
    x,
    y,
    fullmap,
    result,
    temperature=None,
):
    """
    Plot the measured profile, Gaussian fit, and residual.
    """

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 8),
    )

    # ========================================================
    # TOP: Original data and fit
    # ========================================================

    ax[0].plot(
        x,
        y,
        ".",
        markersize=3,
        label="Measured",
    )

    if not result["success"]:

        ax[0].set_title(
            "Fit failed"
        )

    else:

        p = result["fit"]["popt"]

        if result["fit_type"] == "single":

            y_total = gaussian(
                x,
                *p
            )

            ax[0].plot(
                x,
                y_total,
                label="Single Gaussian",
            )

            np.savetxt(f"data_array_{temperature}.csv", y_total)

            ax[0].axvline(
                result["x01"],
                linestyle="--",
                label=(
                    f"x₁ = "
                    f"{result['x01']:.3f} mm"
                ),
            )

        else:

            y_total = double_gaussian(
                x,
                *p
            )
            np.savetxt(f"data_array_{temperature}.csv", y_total)
            # Individual Gaussian components
            y_g1 = (
                p[0]
                + p[1]
                * np.exp(
                    -((x - p[2]) ** 2)
                    / (2 * p[3] ** 2)
                )
            )

            y_g2 = (
                p[0]
                + p[4]
                * np.exp(
                    -((x - p[5]) ** 2)
                    / (2 * p[6] ** 2)
                )
            )

            ax[0].plot(
                x,
                y_total,
                label="Total fit",
            )

            ax[0].plot(
                x,
                y_g1,
                "--",
                label="Peak 1",
            )

            ax[0].plot(
                x,
                y_g2,
                "--",
                label="Peak 2",
            )

            ax[0].axvline(
                result["x01"],
                linestyle=":",
                label=(
                    f"x₁ = "
                    f"{result['x01']:.3f} mm"
                ),
            )

            ax[0].axvline(
                result["x02"],
                linestyle=":",
                label=(
                    f"x₂ = "
                    f"{result['x02']:.3f} mm"
                ),
            )

    if temperature is not None:

        ax[0].set_title(
            f"VIPA Fit — "
            f"T = {temperature:.3f} K"
        )

    ax[0].set_ylabel(
        "Power"
    )

    ax[0].grid(
        alpha=0.3
    )
    ax[0].set_xlabel("stage axis, (mm)")
    ax[0].legend()

    image = ax[1].imshow(fullmap,cmap = "jet", vmin = 0, vmax = 5)
    plt.colorbar(image)

    plt.tight_layout()
    plt.show()

#plot vipa model
def plot_vipa_model(x_relative, results, index=0):

    def vipa_power(theta_l, wavelength_m, r1, r2, theta_air):

        #theta_air = np.deg2rad(DEFAULTS["theta_air_deg"])
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

    result = results[index]

    T = result["temperature_K"]

    wavelength_um = (
        DEFAULTS["lambda_qcl_um"]
        + DEFAULTS["dLambda_dT_nm_K"]
        * (T - 50.0) / 1000.0
    )

    theta_l = np.arctan(
        x_relative * 1e-3 / DEFAULTS["fcam"]
    )

    power = vipa_power(
        theta_l,
        wavelength_um * 1e-6,
        DEFAULTS["r1"],
        DEFAULTS["r1"],
    )

    power /= np.max(power)

    print(f"T = {T:.2f} K")
    print(f"lambda = {wavelength_um:.6f} um")
    print(f"x1 = {result['x01']:.4f} mm")
    print(f"x2 = {result['x02']:.4f} mm")

    plt.figure(figsize=(10, 5))

    plt.plot(
        x_relative,
        power,
        label="VIPA model",
    )

    plt.axvline(
        result["x01"],
        linestyle="--",
        label=f"x1 = {result['x01']:.2f} mm",
    )

    if np.isfinite(result["x02"]):
        plt.axvline(
            result["x02"],
            linestyle="--",
            label=f"x2 = {result['x02']:.2f} mm",
        )

    plt.xlabel("Focal Plane Position Relative to 50 mm (mm)")
    plt.ylabel("Normalized Power")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
#MATCH PEAK
def match_vipa_peak(x_measured, model_peak_angles):

    theta_measured = np.rad2deg(
        np.arctan(
            x_measured * 1e-3
            / DEFAULTS["fcam"]
        )
    )

    i = np.argmin(
        np.abs(
            model_peak_angles
            - theta_measured
        )
    )

    theta_model = model_peak_angles[i]

    x_model = (
        DEFAULTS["fcam"]
        * np.tan(
            np.deg2rad(theta_model)
        )
        * 1000.0
    )

    return theta_model, x_model
   
#FIT AMPS
def fit_vipa_amplitudes(results):
    """
    Fit r1, r2, and a global intensity scale to the
    Gaussian order amplitudes.

    Uses the existing VIPA model without modifying it.
    """

    # --------------------------------------------------------
    # Collect measured amplitudes
    # --------------------------------------------------------

    temperatures = []
    amplitudes = []
    order_numbers = []

    for result in results:

        if not result["success"]:
            continue

        T = result["temperature_K"]

        # First order
        temperatures.append(T)
        amplitudes.append(result["A1"])
        order_numbers.append(1)

        # Second order, if present
        if result["n_peaks"] == 2:

            temperatures.append(T)
            amplitudes.append(result["A2"])
            order_numbers.append(2)

    temperatures = np.asarray(temperatures)
    amplitudes = np.asarray(amplitudes)
    order_numbers = np.asarray(order_numbers)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    def model(params):

        r1, r2, scale = params

        model_amplitudes = []

        for T, order in zip(
            temperatures,
            order_numbers,
        ):

            # Wavelength relative to 50 K
            wavelength_um = (
                DEFAULTS["lambda_qcl_um"]
                + DEFAULTS["dLambda_dT_nm_K"]
                * (T - 50.0)
                / 1000.0
            )

            wavelength_m = (
                wavelength_um * 1e-6
            )

            # ------------------------------------------------
            # Find the measured order position
            # ------------------------------------------------

            result = next(
                r for r in results
                if (
                    r["temperature_K"] == T
                    and r["success"]
                )
            )

            if order == 1:
                x = result["x01"]
            else:
                x = result["x02"]

            # Convert focal-plane position to angle
            theta_l = np.arctan(
                x * 1e-3
                / DEFAULTS["fcam"]
            )

            angle_tot = (
                np.deg2rad(
                    DEFAULTS["theta_air_deg"]
                )
                + theta_l
            )

            # ------------------------------------------------
            # VIPA physics
            # ------------------------------------------------

            sin_t = np.sin(
                angle_tot
            )

            sqrt_t = np.sqrt(
                DEFAULTS["n"]**2
                - sin_t**2
            )

            theta_res = np.arcsin(
                sin_t / DEFAULTS["n"]
            )

            M = np.round(
                DEFAULTS["L"]
                / (
                    2.0
                    * DEFAULTS["h"]
                    * np.tan(theta_res)
                )
            )

            kvec = (
                2.0 * np.pi
                / wavelength_m
            )

            delta = (
                2.0
                * DEFAULTS["h"]
                * kvec
                * sqrt_t
            )

            reflectivity = r1 * r2

            F1 = (
                4.0
                * reflectivity
                / (
                    1.0
                    - reflectivity
                )**2
            )

            FM = (
                4.0
                * reflectivity**M
                / (
                    1.0
                    - reflectivity**M
                )**2
            )

            P = (
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

            model_amplitudes.append(
                scale * P
            )

        return np.asarray(
            model_amplitudes
        )

    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    def residuals(params):

        return (
            model(params)
            - amplitudes
        )

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    p0 = [
        DEFAULTS["r1"],
        DEFAULTS["r1"],
        np.max(amplitudes),
    ]

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    fit = least_squares(
        residuals,
        p0,
        bounds=(
            [0.5, 0.5, 0.0],
            [0.9999, 0.9999, np.inf],
        ),
    )

    r1_fit = fit.x[0]
    r2_fit = fit.x[1]
    scale_fit = fit.x[2]

    return {
        "success": fit.success,
        "r1": r1_fit,
        "r2": r2_fit,
        "scale": scale_fit,
        "result": fit,
        "temperatures": temperatures,
        "amplitudes": amplitudes,
        "order_numbers": order_numbers,
        "model_amplitudes": model(fit.x),
    }



if __name__ == "__main__":
    import os
    VIPA_AXIS_MM = 0.0


    mapPath = ".\\OUTPUT_MAPS"
    maps = []
    temperatures = []
    full_maps = []
    with os.scandir(mapPath) as maps_paths:
        for path in maps_paths:
            fmap = np.loadtxt(f"{mapPath}\\{path.name}", delimiter=",")
            map = np.sum(np.loadtxt(f"{mapPath}\\{path.name}", delimiter=",")[80:160,:], axis = 0)
            temp = float(path.name.split("_")[-1].strip(".csv"))
            maps.append(map)
            temperatures.append(temp)
            full_maps.append(fmap[::-1, :])

    profiles = np.asarray(
        maps
    )

    # --------------------------------------------------------
    # Fit everything
    # --------------------------------------------------------

    x_relative, results = (
        fit_all_profiles(
            profiles,
            full_maps,
            temperatures,
            show_plots=True,
        )
    )
    



