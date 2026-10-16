
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# USER SETTINGS
IMAGE_DIR = Path(r".\\OUTPUT_MAPS")
IMAGE_DIR_flat = Path(r".\DUAL_PROFILES")

# Set to a specific CSV filename to test only one image,
# or leave as None to process every CSV in IMAGE_DIR.
TEST_FILE = None

# Fit bounds and initial guess for r2
R2_INITIAL = np.sqrt(0.9)
R2_BOUNDS = (0.80, 0.98)

# Entrance-angle bounds
THETA_AIR_INITIAL = np.deg2rad(11.23)
THETA_AIR_BOUNDS = (np.deg2rad(10.0), np.deg2rad(13.0))


THETA_L_OFF_INIT = np.deg2rad(1.23)
THETA_L_OFF_BOUNDS = (np.deg2rad(-5.0), np.deg2rad(5.0))

Foffset_INITIAL = 1.0e-3
Foffset_BOUNDS = (-2.0e-3, 2.0e-3)

N_INITIAL = 3.42
N_BOUNDS = (3.41,3.43)


# VIPA DEFAULTS -- PHYSICAL PARAMETERS FIXED

DEFAULTS = {
    "W": 25.4e-3,             # m
    "fcyl": 0.200,            # m
    "n": 3.42,               # Si index at 4 K
    "h": 9.9095e-3,            # m, VIPA thickness
    "theta_air_deg": 11.23,
    "r1": np.sqrt(0.997),
    "L": 29.43e-3,            # m
    "fcam": 0.646,            # m
    "Spix": 0.035,            # mm per pixel
    "lambda_qcl_um": 79.701,  # wavelength at 65 K, um
    "dLambda_dT_nm_K": 2.714, # nm/K
}
#2d FITS

def load_testData(testPath, first_order_peaks, second_order_peaks):

    def temp_dist_conversion(peak_array):
        print(np.array(peak_array))
        peak_vals = []
        for index, val in enumerate(peak_array):
            if val != 0:
                peak_vals.append([(144+index*288), val])

        print(peak_vals)
        return np.array(peak_vals)

    first_order_peaks = temp_dist_conversion(first_order_peaks)
    second_order_peaks = temp_dist_conversion(second_order_peaks)


    with os.scandir(testPath) as maps:
        images = []
        for map in maps:
            image = np.fliplr(np.loadtxt(f"{testPath}\\{map.name}", delimiter=","))
            images.append(image)

        #stack maps
        stack_base = images[0]
        stacked_map = stack_base
        for i in range(1,len(images)):
            stacked_map = np.vstack((stacked_map,images[i]))

    return stacked_map, first_order_peaks, second_order_peaks


# WAVELENGTH CALIBRATION

def wavelength_from_temperature(temperature_K):
    """
    Reference wavelength: 79.701 um at 65 K.
    dLambda/dT = 2.714 nm/K.
    Returns wavelength in meters.
    """
    wavelength_um = (
        DEFAULTS["lambda_qcl_um"]
        + DEFAULTS["dLambda_dT_nm_K"]
        * (temperature_K - 65.0) / 1000.0
    )
    return wavelength_um * 1e-6

def matlab_round(x, decimals=0):
    # Scale to the target decimal place
    factor = 10 ** decimals
    scaled = x * factor
    
    # Extract signs and apply the "away from zero" offset
    rounded = np.trunc(scaled + np.copysign(0.5, scaled))
    
    # Scale back down
    return rounded / factor

# VIPA PHYSICS MODEL -- UNCHANGED

def vipa_power(theta_l, wavelength_m, r1, r2, theta_air, foffset, n, theta_l_offset_rad):
    theta_l_offset_rad = 0
    theta_l = theta_l - theta_l_offset_rad
    def VIPA_powerEnvelope(theta_l, foffset):
        return np.exp(((-2*(DEFAULTS["fcyl"]-foffset)**2 *np.tan(theta_l)**2)/(DEFAULTS["fcam"] - foffset)**2)/ ((DEFAULTS["fcam"]-foffset)**2 * DEFAULTS["W"]**2))
    
    angle_tot = theta_air + theta_l
    thickness = DEFAULTS["h"] / np.cos(theta_air)
    sin_t = np.sin(angle_tot)
    sqrt_t = np.sqrt(n**2 - sin_t**2)
    theta_res = np.arcsin(np.sin(theta_air)/ n)

    M = matlab_round(
        DEFAULTS["L"]
        / (2.0 * thickness * np.tan(theta_res))
    )

    k = 2.0 * np.pi / wavelength_m
    delta = 2.0 * thickness * k * sqrt_t
    R = r1 * r2

    F1 = 4.0 * R / (1.0 - R)**2
    FM = 4.0 * R**M / (1.0 - R**M)**2

    return VIPA_powerEnvelope(theta_l, foffset)*(
            1.0 + FM * np.sin(M * delta / 2.0)**2
        ) / (
            1.0 + F1 * np.sin(delta / 2.0)**2
        )


def vipa_power_LM(x, wavelength_m, r1, r2, theta_i, foffset):
    envelope_portion = np.exp(-(2*DEFAULTS["fcyl"]**2 * x**2)/ ((DEFAULTS["fcam"])**2 * (1e-3)**2))
    k = 2*np.pi / wavelength_m
    thickness_eff = DEFAULTS["h"]/np.cos(theta_i)
    phase_portion = 2*thickness_eff*np.cos(theta_i) - (2*thickness_eff*x)/(DEFAULTS["fcam"] - foffset) - (thickness_eff*np.cos(theta_i)*x**2)/(DEFAULTS["fcam"] - foffset)**2
    lorentzian_portion = 1/((1-r1*r2)**2 + 4*(r1*r2)*(np.sin(k*phase_portion/2)))
    return envelope_portion*lorentzian_portion


# PIXEL-TO-ANGLE CONVERSION 

X_OFFSET_MM = 2.0#0.0

def pixel_angles(n_pixels):
    x_mm = -.04 + np.arange(n_pixels) * DEFAULTS["Spix"]
    x_rel_m = (x_mm - 50.0 - X_OFFSET_MM) * 1e-3
    return np.arctan(x_rel_m / DEFAULTS["fcam"]), x_mm

# LOAD AND PREPARE PROFILES


if TEST_FILE is not None:
    image_paths = [IMAGE_DIR / TEST_FILE]
else:
    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    )
    prof_paths = sorted(
            p for p in IMAGE_DIR_flat.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv"
        )

if not image_paths:
    raise FileNotFoundError(f"No CSV images found in {IMAGE_DIR}")

datasets = []

for image_path, prof_path in zip(image_paths,prof_paths):
    
    temperature_K = float(image_path.stem.split("_")[-1])
    wavelength_m = wavelength_from_temperature(temperature_K)

    
    image =np.fliplr(np.loadtxt(image_path, delimiter=","))[60:180,:]

    
    #profile = np.sum(image, axis=0).astype(float)
    profile = np.sum(image,axis = 0)

    theta_l, x_mm = pixel_angles(profile.size)

    valid = np.isfinite(profile) & np.isfinite(theta_l)
    theta_fit = theta_l[valid]
    y_fit = profile[valid]

    data_max = np.max(y_fit)
    if not np.isfinite(data_max) or data_max <= 0:
        print(f"Skipping {image_path.name}: invalid data maximum.")
        continue

    y_norm = y_fit / data_max

    datasets.append({
        "path": image_path,
        "image": image,
        "temperature_K": temperature_K,
        "wavelength_m": wavelength_m,
        "wavelength_um": wavelength_m * 1e6,
        "x_mm": x_mm,
        "theta_l": theta_fit,
        "profile": profile,
        "valid": valid,
        "y_fit": y_fit,
        "y_norm": y_norm,
        "data_max": data_max,
        "theta_l" : theta_l,

    })

if not datasets:
    raise RuntimeError("No valid profiles available for fitting.")

print(f"Loaded {len(datasets)} profiles for co-fitting.")


# ============================================================
# JOINT FIT
#
# Shared parameters:
#   r2, theta_air
#
# Per-image parameters:
#   normalized background offset
#
# Each image retains its own known wavelength and data_max.
# ============================================================

n_datasets = len(datasets)


def joint_model(xdata, r2, theta_air,foffset, n, theta_l_offset,  *offsets, ):
    """
    xdata is a concatenated array of theta values.
    Split it back into the individual image profiles.
    """
    predictions = []
    start = 0

    for i, dataset in enumerate(datasets):
        n_points = len(dataset["theta_l"])
        theta = xdata[start:start + n_points]
        start += n_points

        power = vipa_power(
            theta,
            dataset["wavelength_m"],
            DEFAULTS["r1"],
            r2,
            theta_air,
            foffset,
            n,
            theta_l_offset,
        )

        # Preserve per-profile model normalization.
        pmax = np.max(power)
        if not np.isfinite(pmax) or pmax <= 0:
            raise ValueError("Invalid VIPA power normalization.")

        power_norm = power / pmax
        predictions.append(power_norm + offsets[i])

    return np.concatenate(predictions)

def joint_model_LM(xdata, r2, theta_air,foffset, *offsets, ):
    """
    xdata is a concatenated array of theta values.
    Split it back into the individual image profiles.
    """
    predictions = []
    start = 0

    for i, dataset in enumerate(datasets):
        n_points = len(dataset["theta_l"])
        theta = xdata[start:start + n_points]
        start += n_points

        power = vipa_power_LM(
            xdata,
            dataset["wavelength_m"],
            DEFAULTS["r1"],
            r2,
            theta_air,
            foffset,
        )

        # Preserve per-profile model normalization.
        pmax = np.max(power)
        if not np.isfinite(pmax) or pmax <= 0:
            raise ValueError("Invalid VIPA power normalization.")

        power_norm = power / pmax
        predictions.append(power_norm + offsets[i])

    return np.concatenate(predictions)


# Concat all measured angular coordinates and normalized profiles.
theta_all = np.concatenate([d["theta_l"] for d in datasets])
y_all = np.concatenate([d["y_norm"] for d in datasets])

# Initial offset for each image.
offset_initial = [
    0.0 for _ in datasets
]
mode = "n - dep"
if mode == "n - dep":

    p0 = [
        R2_INITIAL,
        THETA_AIR_INITIAL,
        Foffset_INITIAL,
        N_INITIAL,
        THETA_L_OFF_INIT,
        *offset_initial,
        
    ]

    lower_bounds = [
        R2_BOUNDS[0],
        THETA_AIR_BOUNDS[0],
        Foffset_BOUNDS[0],
        N_BOUNDS[0],
        THETA_L_OFF_BOUNDS[0],
        *([-np.inf] * n_datasets),
        
    ]

    upper_bounds = [
        R2_BOUNDS[1],
        THETA_AIR_BOUNDS[1],
        Foffset_BOUNDS[1],
        N_BOUNDS[1],
        THETA_L_OFF_BOUNDS[1],
        *([np.inf] * n_datasets),
        
    ]

    params, covariance = curve_fit(
        joint_model,
        theta_all,
        y_all,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=100000,
    )

    r2_fit = params[0]
    theta_air_fit = params[1]
    foffset_fit = params[2]
    offsets_fit = params[5:]
    n_fit = params[3]
    theta_offset_l_fit = params[4]

    print("\n" + "=" * 60)
    print("JOINT FIT RESULTS")
    print("=" * 60)
    print(f"Shared fitted r2: {r2_fit:.8f}")
    print(
        "Shared entrance angle: "
        f"{np.rad2deg(theta_air_fit):.6f} deg"
    )
    print(f"Shared focus (out) offset : {foffset_fit} m")
    print(f"shared refractive index of silicon substrate: {n_fit}")



if mode == "LM":
    p0 = [
            R2_INITIAL,
            THETA_AIR_INITIAL,
            Foffset_INITIAL,
            N_INITIAL,
            *offset_initial,
            
        ]
    
    lower_bounds = [
        R2_BOUNDS[0],
        THETA_AIR_BOUNDS[0],
        Foffset_BOUNDS[0],
        N_BOUNDS[0],
        *([-np.inf] * n_datasets),
        
    ]

    upper_bounds = [
        R2_BOUNDS[1],
        THETA_AIR_BOUNDS[1],
        Foffset_BOUNDS[1],
        N_BOUNDS[1],
        *([np.inf] * n_datasets),
        
    ]

    params, covariance = curve_fit(
        joint_model,
        theta_all,
        y_all,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=100000,
    )

    r2_fit = params[0]
    theta_air_fit = params[1]
    foffset_fit = params[2]
    offsets_fit = params[3:]
    

    print("\n" + "=" * 60)
    print("JOINT FIT RESULTS")
    print("=" * 60)
    print(f"Shared fitted r2: {r2_fit:.8f}")
    print(
        "Shared entrance angle: "
        f"{np.rad2deg(theta_air_fit):.6f} deg"
    )
    print(f"Shared focus (out) offset : {foffset_fit} m")
    

# EVALUATE AND PLOT EACH PROFILE
temps = []
fitted_profs = []
xdatas = []
first_order_peaks = []
second_order_peaks = []
wavelengths = []
for i, dataset in enumerate(datasets):
    theta_fit = dataset["theta_l"]
    data_max = dataset["data_max"]
    y_fit = dataset["y_fit"]
    offset_norm = offsets_fit[i]
    if mode == "n - dep":
        power = vipa_power(
            theta_fit,
            dataset["wavelength_m"],
            DEFAULTS["r1"],
            r2_fit,
            theta_air_fit,
            foffset_fit,
            n_fit,
            theta_offset_l_fit
        )
    elif mode == "LM":
        power = vipa_power_LM(
                    theta_fit,
                    dataset["wavelength_m"],
                    DEFAULTS["r1"],
                    r2_fit,
                    theta_air_fit,
                    foffset_fit,
        
                )
    power_norm = power / np.max(power)

    fitted_norm = power_norm + offset_norm
    fitted_profile = np.full(dataset["profile"].shape, np.nan)
    fitted_profile[dataset["valid"]] = fitted_norm * data_max

    residuals = y_fit - fitted_norm * data_max
    rmse = np.sqrt(np.mean(residuals**2))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r_squared = (
        1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    )

    dataset["fit"] = fitted_profile
    dataset["r_squared"] = r_squared
    dataset["rmse"] = rmse
    dataset["offset"] = offset_norm * data_max
    
    print(f"\n{dataset['path'].name}")
    print(f"  Temperature: {dataset['temperature_K']:.2f} K")
    print(f"  Wavelength:  {dataset['wavelength_um']:.6f} um")
    print(f"  Data max:    {data_max:.6g}")
    print(f"  Offset:      {dataset['offset']:.6g}")
    

    fig, (ax_profile, ax_image) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )

    ax_profile.plot(
        dataset["x_mm"],
        dataset["profile"],
        label="Measured vertical sum",
        linewidth=1.2,
    )

    peaks, props = find_peaks(dataset["fit"],10)
    print(f"peaks: {peaks}")



    if len(peaks) == 1:
        first_order_peaks.append(peaks[0])
        second_order_peaks.append(0)
    else:
        first_order_peaks.append(peaks[0])
        second_order_peaks.append(peaks[1])



    ax_profile.plot(
        dataset["x_mm"],
        fitted_profile,
        "--",
        label="Joint VIPA fit",
        linewidth=1.5,
    )

    ax_profile.set_xlabel("Output-plane coordinate (mm)")
    ax_profile.set_ylabel("Vertically summed signal (a.u.)")
    ax_profile.set_title(
        f"{dataset['path'].name} | "
        f"T={dataset['temperature_K']:.2f} K | "
        
    )
    ax_profile.grid(True, alpha=0.3)
    ax_profile.legend()

    ax_image.imshow(
        dataset["image"],
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )
    ax_image.set_xlabel("Pixel column")
    ax_image.set_ylabel("Pixel row")
    ax_image.set_title("Flipped original image")
    temps.append(dataset["temperature_K"])
    fitted_profs.append(fitted_profile)
    xdatas.append(dataset["x_mm"])
    wavelengths.append(dataset["wavelength_um"])
##MASTER PLOT
fig, ax = plt.subplots()
master_envelope = np.exp(((-2*(DEFAULTS["fcyl"]-foffset_fit)**2 *np.tan(theta_l)**2)/(DEFAULTS["fcam"] - foffset_fit)**2)/ ((DEFAULTS["fcam"]-foffset_fit)**2 * DEFAULTS["W"]**2))
scale = np.max(np.array(fitted_profs))/np.max(master_envelope)
ax.plot(dataset["x_mm"], scale*master_envelope)
for temp,xdata,data in zip(temps, xdatas,fitted_profs):
    ax.plot(xdata, data, label = f"QCL Temp: {temp} K")
    
ax.legend()

fig,ax = plt.subplots()
fullim, fop, sop = load_testData(".\\OUTPUT_MAPS", first_order_peaks, second_order_peaks)

ax.imshow(fullim, cmap = "jet", vmin = 0, vmax = np.max(fullim), origin="lower",
    aspect="auto",
    extent= [x_mm[0], x_mm[-1],0,288*35*fullim.shape[0]/1000],
    interpolation="nearest")




# Primary axis: VIPA model fits in physical coordinates
axis_o1 =  -.04+fop[:, 1]*DEFAULTS["Spix"]
axis_o2 =  -.04+sop[:, 1]*DEFAULTS["Spix"]
ax.plot(axis_o1, fop[:, 0]*35*fullim.shape[0]/1000 /11, color="red", linestyle="--", label="O1 FIT")
ax.plot(axis_o2, sop[:, 0]*35*fullim.shape[0]/1000 / 11, color="orange", linestyle="--", label="O2 FIT")

ax.set_xlabel("Focal plane (mm)")
ax.set_ylabel("Wavelength (um)")
ax.set_title("VIPA FIT")

# Secondary axis: independent image-row coordinates
n_maps = len(wavelengths)
rows_per_map = image.shape[0] / n_maps

y_centers = (np.arange(n_maps) + 0.5) * rows_per_map

ax_wave = ax.twinx()

# Independent limits: image rows, not the primary wavelength limits
ax_wave.set_ylim(0, image.shape[0])
ax_wave.invert_yaxis()  # Remove this line if row 0 is at the bottom

ax_wave.set_yticks(y_centers)
ax_wave.set_yticklabels([f"{w:.4f}" for w in wavelengths])
ax_wave.set_ylabel("Wavelength (um)")

ax.legend(loc="upper right")
ax.set_title("VIPA FIT")

# Hide the primary (left) y-axis
ax.tick_params(axis="y", left=False, labelleft=False)
ax.spines["left"].set_visible(False)

# Keep the independent right-hand wavelength axis visible
ax_wave.tick_params(axis="y", right=True, labelright=True)
ax_wave.spines["right"].set_visible(True)


ax.legend()
# JOINT FIT SUMMARY

print("\n" + "=" * 60)
print("CO-FIT COMPLETE")
print("=" * 60)
print(f"Profiles fitted: {len(datasets)}")
print(f"Shared r2: {r2_fit:.8f}")
print(f"Shared entrance angle: {np.rad2deg(theta_air_fit):.6f} deg")
if mode != "LM":
    print(f"Shared refractive index: {n_fit}")
print(f"Shared focus offset: {foffset_fit} m")

if covariance is not None and np.all(np.isfinite(covariance)):
    parameter_errors = np.sqrt(np.diag(covariance))
    print(f"Estimated r2 uncertainty: {parameter_errors[0]:.3g}")
    print(
        "Estimated entrance-angle uncertainty: "
        f"{np.rad2deg(parameter_errors[1]):.3g} deg"
    )
else:
    print("Parameter covariance unavailable or poorly conditioned.")
plt.show()

def wavelength_old_reference(temperature_K):
    return (
        DEFAULTS["lambda_qcl_um"]
        + DEFAULTS["dLambda_dT_nm_K"]
        * (temperature_K - 50.0) / 1000.0
    ) * 1e-6

def VIPA_powerEnvelope(theta_l, foffset):
    return np.exp((-2*(DEFAULTS["fcyl"]-foffset)**2 *np.tan(theta_l)**2)/(DEFAULTS["fcam"])/ ((DEFAULTS["fcam"]-foffset)**2 * DEFAULTS["W"]**2))

