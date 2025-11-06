"""
LI-600 Stomatal Conductance Correction

This module provides correction functions for systematic positive bias in
stomatal conductance measurements from the LI-600 porometer, following the
method of Rizzo & Bailey (2025).

The correction addresses temperature variations in the instrument's flow path
that cause systematic measurement errors, particularly at high conductance
values and low humidity conditions.

Reference:
    Rizzo, K.T. & Bailey, B.N. (2025). A psychrometric temperature correction
    for porometer measurements of stomatal conductance. (In review)

Example:
    >>> from phytorch.utils import correct_LI600
    >>> corrected_data = correct_LI600('li600_data.csv', stomatal_sidedness=1)
    >>> print(corrected_data[['gsw', 'gsw_corrected']])
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import os
from typing import Union, Tuple
import warnings


def correct_LI600(
    filepath: Union[str, pd.DataFrame],
    stomatal_sidedness: float = 1.0,
    thermal_conductance: float = 0.007,
    save_output: bool = True,
    output_path: str = None
) -> pd.DataFrame:
    """
    Apply Rizzo & Bailey (2025) correction to LI-600 porometer measurements.

    Corrects systematic bias in stomatal conductance (gsw) and chamber air
    temperature measurements from the LI-600 porometer. The correction solves
    a coupled system of thermodynamic equations accounting for heat transfer
    in the instrument's flow path.

    Parameters
    ----------
    filepath : str or pd.DataFrame
        Path to CSV file exported from LI-600, or DataFrame with LI-600 data.
        Required columns: 'gsw', 'Tref', 'Tleaf', 'rh_r', 'rh_s', 'flow',
        'P_atm', 'E_apparent'
    stomatal_sidedness : float, optional
        Stomatal sidedness correction factor (default: 1.0)
        - 1.0 for hypostomatous (stomata on lower leaf surface only)
        - 2.0 for amphistomatous (stomata on both surfaces equally)
        - Values between 1.0 and 2.0 for intermediate cases
    thermal_conductance : float, optional
        Thermal conductance C in W/°C (default: 0.007)
        This value was empirically calibrated and may vary between instruments
    save_output : bool, optional
        Whether to save corrected data to CSV file (default: True)
    output_path : str, optional
        Output file path. If None, appends '_corrected' to input filename

    Returns
    -------
    pd.DataFrame
        DataFrame with original data plus corrected columns:
        - 'gsw_corrected': Corrected stomatal conductance (mol m⁻² s⁻¹)
        - 'Ta_chamb_corrected': Corrected chamber temperature (°C)
        - 'T_in_corrected': Inlet temperature (°C)
        - 'T_out_corrected': Outlet temperature (°C)
        - 'W_chamb_corrected': Corrected chamber water vapor mole fraction
        - 'stomatal_sidedness': Applied sidedness value

    Notes
    -----
    The correction is based on solving a system of three coupled equations:
    1. Fick's law for water vapor diffusion
    2. Mass balance for water vapor
    3. Energy balance including latent and sensible heat

    The correction is most significant for:
    - High stomatal conductance values (>0.3 mol m⁻² s⁻¹)
    - Low humidity conditions
    - Large leaf-air temperature differences

    If the numerical solver fails to converge for a data point, that point
    is assigned zero values for corrected parameters (indicates bad data).

    Examples
    --------
    Correct hypostomatous leaf measurements:

    >>> data = correct_LI600('measurements.csv', stomatal_sidedness=1.0)

    Correct amphistomatous leaf without saving output:

    >>> data = correct_LI600('data.csv', stomatal_sidedness=2.0, save_output=False)

    Use custom thermal conductance:

    >>> data = correct_LI600('data.csv', thermal_conductance=0.008)

    References
    ----------
    Rizzo, K.T. & Bailey, B.N. (2025). A psychrometric temperature correction
    for porometer measurements of stomatal conductance. (In review)
    """

    # Read or validate input data
    if isinstance(filepath, str):
        data = _read_li600_csv(filepath)
        input_filepath = filepath
    elif isinstance(filepath, pd.DataFrame):
        data = filepath.copy()
        input_filepath = None
    else:
        raise TypeError("filepath must be a string path or pandas DataFrame")

    # Validate required columns
    required_cols = ['gsw', 'Tref', 'Tleaf', 'rh_r', 'rh_s', 'flow', 'P_atm', 'E_apparent']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Initialize result arrays
    n = len(data)
    sidedness = np.full(n, stomatal_sidedness)
    T_ins = np.zeros(n)
    T_chambs = np.zeros(n)
    T_outs = np.zeros(n)
    W_chambs = np.zeros(n)
    gsw_corrected = np.zeros(n)

    # Thermodynamic constants
    a = 0.61365      # Saturation vapor pressure magnitude (kPa)
    b = 17.502       # Saturation vapor pressure slope (dimensionless)
    c = 240.97       # Saturation vapor pressure offset (°C)
    C = thermal_conductance  # Thermal conductance (W/°C = J/(s·°C))

    cpa = 29.14      # Air heat capacity (J/(mol·°C))
    cpw = 33.5       # Water vapor heat capacity (J/(mol·°C))
    lambdaw = 45502  # Water latent heat of vaporization (J/mol)

    s = 0.441786 * 0.01**2  # Leaf area (m²)
    gbw = 2.921              # Boundary layer conductance (mol/(m²·s))

    def es(T):
        """Saturation vapor pressure (kPa) at temperature T (°C)."""
        return a * np.exp(b * T / (T + c))

    def W(T, RH, P_atm):
        """Water vapor mole fraction (mol/mol) from T, RH, and pressure."""
        return es(T) * RH / P_atm

    def h(T, RH, P_atm):
        """Moist air enthalpy (J/mol) at given conditions."""
        W_val = W(T, RH, P_atm)
        return (1.0 - W_val) * cpa * T + W_val * (lambdaw + cpw * T)

    # Track failed convergences
    failed_points = []

    # Process each measurement
    for i in range(n):
        # Extract input values
        T_in = data['Tref'].iloc[i]     # Chamber inlet temp (°C)
        T_leaf = data['Tleaf'].iloc[i]  # Leaf temperature (°C)
        RH_in = data['rh_r'].iloc[i] / 100.0   # Inlet RH (fraction)
        RH_out = data['rh_s'].iloc[i] / 100.0  # Outlet RH (fraction)
        u_in = data['flow'].iloc[i] * 1e-6     # Inlet flow (mol/s)
        P_atm = data['P_atm'].iloc[i]          # Atmospheric pressure (kPa)

        # Initial guesses for solver
        initial_guesses = [
            data['Tref'].iloc[i] - 0.1,    # T_out slightly less than T_in
            data['E_apparent'].iloc[i],    # E from instrument
            data['gsw'].iloc[i] * 0.75     # gsw slightly less than measured
        ]

        def equations(vars):
            """
            System of equations to solve for T_out, E, and gsw.

            Based on Equations 14-16 from Rizzo & Bailey (2025):
            1. Fick's law: E = gtw * (W_leaf - W_chamb)
            2. Mass balance: E = (u_in / s) * (W_out - W_in) / (1 - W_out)
            3. Energy balance: E = (1/s) * ((Q + u_in*h_in)/h_out - u_in)
            """
            T_out, E, gsw = vars

            # Chamber conditions (average of inlet and outlet)
            T_chamb = 0.5 * (T_in + T_out)
            RH_chamb = 0.5 * (RH_in + RH_out)

            # Water vapor mole fractions
            W_chamb = W(T_chamb, RH_chamb, P_atm)
            W_in = W(T_in, RH_in, P_atm)
            W_out = W(T_in, RH_out, P_atm)  # Diffused to T_in
            W_leaf = W(T_leaf, 1.0, P_atm)  # Saturated at leaf surface

            # Enthalpies
            h_in = h(T_in, RH_in, P_atm)
            h_out = h(T_in, RH_out, P_atm)

            # Heat transfer from air to chamber
            Q = C * (T_in - T_chamb)

            # Total (stomatal + boundary layer) conductance
            gtw = (gsw * gbw) / (gsw + gbw)

            # System of three equations
            eq1 = E - gtw * (W_leaf - W_chamb)
            eq2 = E - s**(-1) * u_in * (W_out - W_in) / (1.0 - W_out)
            eq3 = E - s**(-1) * ((Q + u_in * h_in) / h_out - u_in)

            return [eq1, eq2, eq3]

        try:
            # Solve the system
            solution = fsolve(equations, initial_guesses, full_output=True)
            T_out_sol, E_sol, gsw_sol = solution[0]
            info = solution[1]

            # Check convergence quality
            residual = np.sqrt(info['fvec'].dot(info['fvec']))
            if residual > 1e-6:
                # Poor convergence - mark as failed
                T_out_sol = 0
                E_sol = 0
                gsw_sol = 0
                failed_points.append(i)
        except:
            # Solver failure - use zeros
            T_out_sol = 0
            E_sol = 0
            gsw_sol = 0
            failed_points.append(i)

        # Store results
        gsw_corrected[i] = gsw_sol * sidedness[i]
        T_outs[i] = T_out_sol
        T_ins[i] = T_in
        T_chambs[i] = 0.5 * (T_in + T_out_sol)

        # Calculate corrected chamber water vapor mole fraction
        RH_chamb = 0.5 * (RH_in + RH_out)
        W_chambs[i] = W(T_chambs[i], RH_chamb, P_atm)

    # Warn if any points failed
    if failed_points:
        warnings.warn(
            f"Solver failed to converge for {len(failed_points)} data points "
            f"(indices: {failed_points[:10]}{'...' if len(failed_points) > 10 else ''}). "
            f"These points have been set to zero in corrected columns.",
            UserWarning
        )

    # Add corrected values to DataFrame
    data['gsw_corrected'] = gsw_corrected
    data['T_in_corrected'] = T_ins
    data['Ta_chamb_corrected'] = T_chambs
    data['T_out_corrected'] = T_outs
    data['W_chamb_corrected'] = W_chambs
    data['stomatal_sidedness'] = sidedness

    # Save output if requested
    if save_output and input_filepath is not None:
        if output_path is None:
            path, filename = os.path.split(input_filepath)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(path, f"{name}_corrected{ext}")

        data.to_csv(output_path, index=False)
        print(f"Corrected data saved to: {output_path}")

    return data


def _read_li600_csv(filepath: str) -> pd.DataFrame:
    """
    Read CSV file from LI-600 with flexible parsing.

    Handles different LI-600 export formats:
    - Standard format with group/name/unit rows
    - Simplified format with header only
    - Plain CSV format
    """
    try:
        # Try LI-600 format (row 0 = groups, row 1 = names, row 2 = units)
        data = pd.read_csv(filepath, skiprows=[0, 2], header=0)
        _ = data['gsw']  # Validate
        return data
    except (KeyError, pd.errors.ParserError):
        try:
            # Try skipping first 2 rows only
            data = pd.read_csv(filepath, skiprows=2, header=0)
            _ = data['gsw']
            return data
        except:
            try:
                # Try standard CSV
                data = pd.read_csv(filepath)
                _ = data['gsw']
                return data
            except Exception as e:
                raise ValueError(
                    f"Unable to read LI-600 data from {filepath}. "
                    f"Ensure file contains required column 'gsw'. Error: {e}"
                )


def plot_correction(
    data: pd.DataFrame,
    save_path: str = None,
    show: bool = True
) -> Tuple:
    """
    Plot comparison of original vs corrected gsw and W_chamb values.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with both original and corrected values (output of correct_LI600)
    save_path : str, optional
        Path to save figure. If None, generates name from input data
    show : bool, optional
        Whether to display the plot (default: True)

    Returns
    -------
    tuple
        (figure, (ax1, ax2)) matplotlib figure and axes objects

    Examples
    --------
    >>> corrected_data = correct_LI600('data.csv')
    >>> fig, axes = plot_correction(corrected_data, save_path='correction_plot.png')
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    def linear_fit(x, a, c):
        """Linear function for fitting."""
        return a * x + c

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: gsw correction
    x1 = data['gsw'].values
    y1 = data['gsw_corrected'].values

    # Remove NaN/inf values
    mask1 = np.isfinite(x1) & np.isfinite(y1) & (x1 != 0) & (y1 != 0)
    x1_clean = x1[mask1]
    y1_clean = y1[mask1]

    # Plot 1:1 line and corrected values
    ax1.scatter(x1_clean, x1_clean, c='gray', alpha=0.5, s=30, label='Original $g_{sw}$')
    ax1.scatter(x1_clean, y1_clean, c='black', s=50, edgecolors='k',
                linewidth=0.5, alpha=0.7, label='Corrected $g_{sw}$')

    # Fit and plot linear model
    if len(x1_clean) > 1:
        try:
            popt1, _ = curve_fit(linear_fit, x1_clean, y1_clean)
            a1, c1 = popt1
            x_fit1 = np.linspace(x1_clean.min(), x1_clean.max(), 100)
            y_fit1 = linear_fit(x_fit1, a1, c1)
            ax1.plot(x_fit1, y_fit1, 'r-', linewidth=2, alpha=0.7,
                    label=f'Fit: y = {a1:.3f}x + {c1:.4f}')
        except:
            pass

    ax1.set_xlabel('Original $g_{sw}$ (mol m$^{-2}$ s$^{-1}$)', fontsize=13)
    ax1.set_ylabel('$g_{sw}$ (mol m$^{-2}$ s$^{-1}$)', fontsize=13)
    ax1.set_title('LI-600 Stomatal Conductance Correction', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Plot 2: W_chamb correction
    # Calculate original W_chamb
    a = 0.61365
    b = 17.502
    c = 240.97

    def es(T):
        return a * np.exp(b * T / (T + c))

    def W(T, RH, P_atm):
        return es(T) * RH / P_atm

    x2 = W(data['Tref'].values, data['rh_r'].values/100.0, data['P_atm'].values)
    y2 = data['W_chamb_corrected'].values

    # Remove NaN/inf values
    mask2 = np.isfinite(x2) & np.isfinite(y2) & (x2 != 0) & (y2 != 0)
    x2_clean = x2[mask2]
    y2_clean = y2[mask2]

    # Plot 1:1 line and corrected values
    ax2.scatter(x2_clean, x2_clean, c='gray', alpha=0.5, s=30, label='Original $W_{chamb}$')
    ax2.scatter(x2_clean, y2_clean, c='black', s=50, edgecolors='k',
                linewidth=0.5, alpha=0.7, label='Corrected $W_{chamb}$')

    # Fit and plot linear model
    if len(x2_clean) > 1:
        try:
            popt2, _ = curve_fit(linear_fit, x2_clean, y2_clean)
            a2, c2 = popt2
            x_fit2 = np.linspace(x2_clean.min(), x2_clean.max(), 100)
            y_fit2 = linear_fit(x_fit2, a2, c2)
            ax2.plot(x_fit2, y_fit2, 'r-', linewidth=2, alpha=0.7,
                    label=f'Fit: y = {a2:.3f}x + {c2:.6f}')
        except:
            pass

    ax2.set_xlabel('Original $W_{chamb}$ (mol mol$^{-1}$)', fontsize=13)
    ax2.set_ylabel('$W_{chamb}$ (mol mol$^{-1}$)', fontsize=13)
    ax2.set_title('LI-600 Chamber Water Vapor Correction', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig, (ax1, ax2)
