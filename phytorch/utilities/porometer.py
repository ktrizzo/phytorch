"""
Porometer utility functions for computing photosynthesis from conductance measurements.

Functions for solving coupled photosynthesis-stomatal conductance equations
given porometer measurements and FvCB model parameters.
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, brentq
from typing import Dict, Union, Tuple
import warnings


def solve_for_A(
    data: Union[pd.DataFrame, Dict],
    parameters: Dict,
    Ca: float = 420.0,
    use_corrected_gsw: bool = True,
    Q_field: str = 'Qamb',
    T_field: str = 'Tleaf',
    gsw_field: str = 'gsw',
    Q_conversion: float = 0.85
) -> pd.DataFrame:
    """
    Solve for intercellular CO2 (Ci) and net photosynthesis (A) from porometer data.

    Given stomatal conductance measurements and FvCB model parameters, this function
    solves the coupled supply-demand equation:
        A(Ci) = gs * (Ca - Ci)
    where A(Ci) is computed from the FvCB biochemical model.

    Parameters
    ----------
    data : pd.DataFrame or dict
        Porometer measurement data containing:
        - 'gsw' or 'gsw_corrected': Stomatal conductance to water vapor (mol m⁻² s⁻¹)
        - 'Qamb' (or Q_field): Incident PPFD (μmol m⁻² s⁻¹)
        - 'Tleaf' (or T_field): Leaf temperature (°C)
    parameters : dict
        FvCB model parameters:
        - 'Vcmax_25': Maximum Rubisco carboxylation rate at 25°C
        - 'Jmax_25': Maximum electron transport rate at 25°C
        - 'Rd_25': Dark respiration at 25°C (optional, default 1.5)
        - 'alpha': Quantum yield of electron transport (optional, default 0.85)
        - 'theta': Curvature of light response (optional, default 0.7)
        - 'Vcmax_dHa': Vcmax activation energy (kJ/mol, optional, default 73.0)
        - 'Vcmax_dHd': Vcmax deactivation energy (kJ/mol, optional, default 200.0)
        - 'Vcmax_Topt': Vcmax optimum temperature (K, optional, default 311.15)
        - Similar parameters for Jmax (defaults: dHa=33, dHd=200, Topt=311.15)
        - 'Kc_25', 'Ko_25', 'Gamma_25': Michaelis constants (optional, use defaults)
        - 'O': Oxygen concentration (mmol/mol, optional, default 210)
    Ca : float, optional
        Atmospheric CO2 concentration (μmol mol⁻¹, default: 420)
    use_corrected_gsw : bool, optional
        Use 'gsw_corrected' if available, otherwise 'gsw' (default: True)
    Q_field : str, optional
        Name of PPFD field in data (default: 'Qamb')
    T_field : str, optional
        Name of temperature field in data (default: 'Tleaf')
    gsw_field : str, optional
        Name of conductance field if not using corrected (default: 'gsw')
    Q_conversion : float, optional
        Conversion factor for PPFD (e.g., 0.85 for absorbed light, default: 0.85)

    Returns
    -------
    pd.DataFrame
        Original data with added columns:
        - 'Ci': Intercellular CO2 concentration (μmol mol⁻¹)
        - 'A': Net photosynthesis rate (μmol m⁻² s⁻¹)
        - 'Ac': Rubisco-limited rate (μmol m⁻² s⁻¹)
        - 'Aj': RuBP-regeneration-limited rate (μmol m⁻² s⁻¹)

    Notes
    -----
    The function solves for each measurement point independently by finding Ci where:
        FvCB_model(Ci, Q, T) = gs * (Ca - Ci)

    Uses peaked Arrhenius temperature response if dHd and Topt are provided,
    otherwise uses simple Arrhenius response.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'gsw_corrected': [0.1, 0.15, 0.2],
    ...     'Qamb': [1000, 1500, 2000],
    ...     'Tleaf': [25, 27, 30]
    ... })
    >>> params = {
    ...     'Vcmax_25': 100,
    ...     'Jmax_25': 180,
    ...     'Rd_25': 1.5
    ... }
    >>> result = solve_for_A(data, params)
    >>> print(result[['Ci', 'A']])

    References
    ----------
    Farquhar, G.D., von Caemmerer, S., & Berry, J.A. (1980). A biochemical model
    of photosynthetic CO2 assimilation in leaves of C3 species.
    Planta, 149(1), 78-90.
    """

    # Convert to DataFrame if dict
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    else:
        data = data.copy()

    # Determine which gsw field to use
    if use_corrected_gsw and 'gsw_corrected' in data.columns:
        gs_col = 'gsw_corrected'
    else:
        gs_col = gsw_field

    # Extract data
    gs = data[gs_col].values  # mol m⁻² s⁻¹
    Q = data[Q_field].values * Q_conversion  # Convert PPFD
    T = data[T_field].values + 273.15  # Convert to Kelvin

    # Set default parameters
    params = _set_default_parameters(parameters)

    # Pre-compute constants
    R = 0.008314  # kJ/(mol·K), gas constant
    Tref = 298.15  # K, reference temperature

    # Initialize output arrays
    n = len(gs)
    Ci_out = np.zeros(n)
    A_out = np.zeros(n)
    Ac_out = np.zeros(n)
    Aj_out = np.zeros(n)

    # Process each measurement
    failed_points = []

    for i in range(n):
        try:
            # Solve for Ci
            Ci_solved = _solve_ci_at_point(
                gs[i], Q[i], T[i], Ca, params, R, Tref
            )

            # Compute A at solved Ci
            A_val, Ac_val, Aj_val = _compute_photosynthesis(
                Ci_solved, Q[i], T[i], params, R, Tref
            )

            Ci_out[i] = Ci_solved
            A_out[i] = A_val
            Ac_out[i] = Ac_val
            Aj_out[i] = Aj_val

        except Exception as e:
            # If solver fails, set to zero
            Ci_out[i] = 0
            A_out[i] = 0
            Ac_out[i] = 0
            Aj_out[i] = 0
            failed_points.append(i)

    # Warn if any points failed
    if failed_points:
        warnings.warn(
            f"Solver failed to converge for {len(failed_points)} data points "
            f"(indices: {failed_points[:10]}{'...' if len(failed_points) > 10 else ''}). "
            f"These points have been set to zero.",
            UserWarning
        )

    # Add results to DataFrame
    data['Ci'] = Ci_out
    data['A'] = A_out
    data['Ac'] = Ac_out
    data['Aj'] = Aj_out

    return data


def _set_default_parameters(params: Dict) -> Dict:
    """Set default FvCB parameters."""
    defaults = {
        'Rd_25': 1.5,
        'alpha': 0.85,
        'theta': 0.7,
        'Vcmax_dHa': 73.0,
        'Vcmax_dHd': 200.0,
        'Vcmax_Topt': 311.15,
        'Jmax_dHa': 33.0,
        'Jmax_dHd': 200.0,
        'Jmax_Topt': 311.15,
        'Rd_dHa': 46.39,
        'Kc_25': 404.9,
        'Ko_25': 278.4,
        'Gamma_25': 42.75,
        'Kc_dHa': 79.43,
        'Ko_dHa': 36.38,
        'Gamma_dHa': 37.83,
        'O': 210.0  # mmol/mol
    }

    # Update with provided parameters
    p = defaults.copy()
    p.update(params)

    return p


def _temperature_response(T: float, k25: float, dHa: float, dHd: float,
                          Topt: float, R: float, Tref: float) -> float:
    """Peaked Arrhenius temperature response function."""
    # Simple Arrhenius component
    k_arr = k25 * np.exp(dHa / R * (1/Tref - 1/T))

    # If dHd and Topt suggest peaked response
    if dHd > 100 and Topt < 400:  # Reasonable values for deactivation
        dHd_dHa = dHd / dHa
        dHd_dHa = max(dHd_dHa, 1.0001)
        log_term = np.log(dHd_dHa - 1)

        numerator = 1 + np.exp(dHd / R * (1/Topt - 1/Tref) - log_term)
        denominator = 1 + np.exp(dHd / R * (1/Topt - 1/T) - log_term)

        return k_arr * numerator / denominator
    else:
        # Simple Arrhenius only
        return k_arr


def _compute_photosynthesis(Ci: float, Q: float, T: float, params: Dict,
                            R: float, Tref: float) -> Tuple[float, float, float]:
    """
    Compute net photosynthesis and limiting rates at given Ci, Q, T.

    Returns (A, Ac, Aj) where:
    - A: Net photosynthesis (smooth minimum of Ac and Aj)
    - Ac: Rubisco-limited rate
    - Aj: RuBP-regeneration-limited rate
    """
    # Temperature-scaled parameters
    Vcmax = _temperature_response(
        T, params['Vcmax_25'], params['Vcmax_dHa'],
        params['Vcmax_dHd'], params['Vcmax_Topt'], R, Tref
    )

    Jmax = _temperature_response(
        T, params['Jmax_25'], params['Jmax_dHa'],
        params['Jmax_dHd'], params['Jmax_Topt'], R, Tref
    )

    Rd = params['Rd_25'] * np.exp(params['Rd_dHa'] / R * (1/Tref - 1/T))

    # Simple Arrhenius for Michaelis constants
    Kc = params['Kc_25'] * np.exp(params['Kc_dHa'] / R * (1/Tref - 1/T))
    Ko = params['Ko_25'] * np.exp(params['Ko_dHa'] / R * (1/Tref - 1/T))
    Gamma = params['Gamma_25'] * np.exp(params['Gamma_dHa'] / R * (1/Tref - 1/T))

    # Effective Michaelis constant for CO2
    Kco = Kc * (1 + params['O'] / Ko)

    # Electron transport rate (light response)
    theta = max(params['theta'], 0.0001)
    alpha = params['alpha']
    a = theta
    b = -(alpha * Q + Jmax)
    c = alpha * Q * Jmax
    J = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

    # Rubisco-limited rate
    Ac = Vcmax * (Ci - Gamma) / (Ci + Kco) - Rd

    # RuBP-regeneration-limited rate
    Aj = 0.25 * J * (Ci - Gamma) / (Ci + 2*Gamma) - Rd

    # Smooth minimum (hyperbolic)
    theta_smooth = 0.999
    A = (Ac + Aj - np.sqrt((Ac + Aj)**2 - 4*theta_smooth*Ac*Aj)) / (2*theta_smooth)

    return A, Ac, Aj


def _solve_ci_at_point(gs: float, Q: float, T: float, Ca: float,
                       params: Dict, R: float, Tref: float) -> float:
    """
    Solve for Ci at a single measurement point.

    Finds Ci where: A(Ci) = gs * (Ca - Ci)
    """

    def objective(Ci):
        """Equation to solve: A(Ci) - gs*(Ca - Ci) = 0"""
        A, _, _ = _compute_photosynthesis(Ci, Q, T, params, R, Tref)
        supply = gs * (Ca - Ci)
        return A - supply

    # Try bounded solver first (more robust)
    try:
        # Ci must be between Gamma and Ca
        Gamma = params['Gamma_25'] * np.exp(params['Gamma_dHa'] / R * (1/Tref - 1/T))
        Ci_min = max(Gamma * 1.1, 10)  # Slightly above compensation point
        Ci_max = Ca * 0.99  # Slightly below atmospheric

        # Check if bounds bracket the solution
        f_min = objective(Ci_min)
        f_max = objective(Ci_max)

        if f_min * f_max < 0:  # Opposite signs, solution exists
            Ci_solved = brentq(objective, Ci_min, Ci_max, xtol=0.01)
        else:
            # Fall back to fsolve with good initial guess
            Ci_init = Ca * 0.7  # Typical Ci/Ca ratio
            result = fsolve(objective, Ci_init, full_output=True)
            Ci_solved = result[0][0]

            # Check convergence
            if result[2] != 1:  # Did not converge
                raise ValueError("fsolve did not converge")

    except Exception as e:
        # Last resort: try fsolve with multiple initial guesses
        for Ci_init in [Ca * 0.5, Ca * 0.7, Ca * 0.9, 100, 200, 300]:
            try:
                result = fsolve(objective, Ci_init, full_output=True)
                if result[2] == 1:  # Converged
                    Ci_solved = result[0][0]
                    if 0 < Ci_solved < Ca:  # Physically reasonable
                        return Ci_solved
            except:
                continue

        # All attempts failed
        raise ValueError(f"Could not solve for Ci at gs={gs}, Q={Q}, T={T}")

    return Ci_solved
