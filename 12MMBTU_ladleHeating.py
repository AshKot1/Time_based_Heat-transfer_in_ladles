import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- Helper Functions (Define BEFORE use) ---
def K_to_F(T_K):
    if isinstance(T_K, float) and np.isnan(T_K): return np.nan
    return (T_K - 273.15) * 9/5 + 32

# h_in Calculation - USES T_gas_original_K
def calculate_h_in_py(T_gas_K_original, T_wall_inner_K, Re, Pr, gas_k, D_mean, epsilon_gas, epsilon_wall_inner, sigma):
    T_wall_inner_avg = np.mean(T_wall_inner_K) if isinstance(T_wall_inner_K, np.ndarray) else T_wall_inner_K
    if T_wall_inner_avg is None or np.isnan(T_wall_inner_avg): T_wall_inner_avg = 0
    # Convection
    if Pr >= 0.6 and Re > 10000:
        Nu = 0.023 * (Re**0.8) * (Pr**0.4)
        h_conv = Nu * gas_k / D_mean
    else:
        h_conv = 50.0
    # Radiation
    h_rad = 0.0
    deltaT = T_gas_K_original - T_wall_inner_avg
    if abs(deltaT) > 0.01:
        try:
            if T_gas_K_original > 0 and T_wall_inner_avg > 0:
                 h_rad = sigma * epsilon_gas * epsilon_wall_inner * (T_gas_K_original**4 - T_wall_inner_avg**4) / deltaT
            else: h_rad = 0.0
        except (OverflowError, ZeroDivisionError): h_rad = 4*sigma*epsilon_gas*epsilon_wall_inner*T_gas_K_original**3 if T_gas_K_original > 0 else 0.0
    else: h_rad = 4*sigma*epsilon_gas*epsilon_wall_inner*T_gas_K_original**3 if T_gas_K_original > 0 else 0.0
    h_rad = max(0.0, h_rad)
    return h_conv + h_rad

# h_out Calculation - Original
def calculate_h_out_py(T_wall_outer_K, T_ambient_K, epsilon_outer, sigma, L_char): # L_char is H_m here
    T_wall_outer_avg = np.mean(T_wall_outer_K) if isinstance(T_wall_outer_K, np.ndarray) else T_wall_outer_K
    if T_wall_outer_avg is None or np.isnan(T_wall_outer_avg): T_wall_outer_avg = T_ambient_K
    # Natural Convection
    h_conv_nat = 2.0
    deltaT = T_wall_outer_avg - T_ambient_K
    if deltaT > 0 and L_char > 0:
        try:
            h_conv_calc = 1.42 * (deltaT / L_char)**0.25
            h_conv_nat = min(20.0, max(2.0, h_conv_calc))
        except (ValueError, OverflowError): h_conv_nat = 2.0
    # Radiation
    h_rad_out = 0.0
    deltaT_rad = T_wall_outer_avg - T_ambient_K
    if abs(deltaT_rad) > 0.01:
        try:
            if T_wall_outer_avg > 0 and T_ambient_K > 0:
                h_rad_out = sigma * epsilon_outer * (T_wall_outer_avg**4 - T_ambient_K**4) / deltaT_rad
            else: h_rad_out = 0.0
        except (OverflowError, ZeroDivisionError): h_rad_out = 4*sigma*epsilon_outer*T_ambient_K**3 if T_ambient_K > 0 else 0.0
    else: h_rad_out = 4*sigma*epsilon_outer*T_ambient_K**3 if T_ambient_K > 0 else 0.0
    h_rad_out = max(0.0, h_rad_out)
    return h_conv_nat + h_rad_out

# --- Parameters ---
params = {
  # --- Temperatures ---
  "T_wall_start_K": 1005.3722222222221,       # USER INPUT - Change according to requirements (Initial Wall Temp K)
  "T_wall_target1_K": 1372.0388888888888,     # USER INPUT - Change according to requirements (~2010 F)
  "T_wall_target2_K": 1422.0388888888888,     # USER INPUT - Change according to requirements (~2100 F)
  "T_gas_original_K": 1258.8249999999998,     # USER INPUT - Change according to requirements (~1806 F, used for sensible heat)
  "T_ambient_K": 298.15,                      # USER INPUT - Change according to requirements (Ambient Temp K)
  # --- Geometry & Mass ---
  "A_inner_total_m2": 34.98648712421236,      # Derived from user geometry inputs
  "A_outer_total_m2": 42.95377026546754,      # Derived from user geometry inputs
  "Mass_refractory_kg": 26094.113670727387,    # Derived from user geometry inputs
  "thickness_m": 0.2286,                     # Derived from user geometry inputs
  "D_inner_mean_m": 2.2885400000000002,     # Derived from user geometry inputs
  "H_m": 4.39674,                           # Derived from user geometry inputs (Height)
  # --- Material Properties ---
  "Cp_refractory_J_kgK": 910.0,              # USER INPUT - Change according to requirements (Refractory Cp)
  "k_refractory_W_mK": 1.75,               # USER INPUT - Change according to requirements (Refractory k)
  "rho_refractory_kgm3": 2872.0,            # USER INPUT - Change according to requirements (Refractory Density)
  "epsilon_wall_inner": 0.8,               # ASSUMPTION - Change according to requirements (Inner Emissivity)
  "epsilon_wall_outer": 0.7,               # ASSUMPTION - Change according to requirements (Outer Emissivity)
  # --- Gas Properties (from user input at specific conditions) ---
  "gas_k_W_mK": 0.0454,                   # USER INPUT - Change according to requirements (Gas Thermal Cond.)
  "Pr": 0.5369252775330396,                # USER INPUT - Change according to requirements (Gas Prandtl No.)
  "Re": 5058684.615348837,                 # USER INPUT - Change according to requirements (Gas Reynolds No. based on inlet)
  "epsilon_gas_est": 0.15,                # ASSUMPTION - Change according to requirements (Estimated Gas Emissivity)
  # --- Constants ---
  "sigma": 5.67e-08                         # Physical Constant
}

# --- Post-Combustion Heat Input ---
# Value from Block 1 - requires assumption on inlet area.
# Example value if Block 1 not run in same session:
try:
  Q_post_comb_W = _Q_post_comb_W if '_Q_post_comb_W' in locals() else 663221.1 # Approx 6.6 MW from 0.5m assumption
except NameError:
  Q_post_comb_W = 663221.1 # Default if _Q_post_comb_W isn't defined
print(f"--- Using Post-Combustion Heat: {Q_post_comb_W / 1e6:.3f} MW (Requires Inlet Area Assumption) ---")
print(f"--- Using T_gas_original: {params['T_gas_original_K']:.1f} K for sensible heat calc ---")
print(f"--- Using Original Heat Loss Calculation ---")

# --- Simulation ---
dt_seconds = 60.0
t_seconds = 0.0
T_wall_K = params['T_wall_start_K']

# Correctly initialize lists to be appended to
results_lumped = {
    "Time_min": [], "T_wall_K": [], "T_wall_F": [],
    "h_in": [], "Q_in_sensible_kW": [], "Q_in_post_comb_kW": [], # Corrected: Append later
    "h_out": [], "Q_out_kW": [], "dTdt_K_min": [],
}

time_target1_min = None
time_target2_min = None

print("\nStarting Lumped Capacitance simulation (Post-Combustion, Corrected)...")
start_sim_time = time.time()

# Loop until the HIGHER target is reached
while T_wall_K < params['T_wall_target2_K']:
    time_minutes = t_seconds / 60.0

    if time_minutes > 240: print(f"Warning: Simulation stopped after 240 minutes."); break
    if dt_seconds <= 0: break # Safety check

    # Calculate HTCs and Sensible Heat Rate
    h_in = calculate_h_in_py(
        params['T_gas_original_K'], T_wall_K, params['Re'], params['Pr'], params['gas_k_W_mK'],
        params['D_inner_mean_m'], params['epsilon_gas_est'], params['epsilon_wall_inner'], params['sigma']
    )
    h_out = calculate_h_out_py(
        T_wall_K, params['T_ambient_K'], params['epsilon_wall_outer'],
        params['sigma'], params['H_m']
    )

    Q_in_sensible = h_in * params['A_inner_total_m2'] * (params['T_gas_original_K'] - T_wall_K)
    Q_out = h_out * params['A_outer_total_m2'] * (T_wall_K - params['T_ambient_K'])
    Q_out = max(0.0, Q_out)

    try:
        if params['Mass_refractory_kg'] <= 0 or params['Cp_refractory_J_kgK'] <= 0: raise ValueError("Mass or Cp invalid.")
        dTdt_K_sec = (Q_in_sensible + Q_post_comb_W - Q_out) / (params['Mass_refractory_kg'] * params['Cp_refractory_J_kgK'])
    except ValueError as e:
        print(f"Error calculating dTdt: {e}")
        dTdt_K_sec = 0

    # --- Log results BEFORE updating temperature ---
    results_lumped["Time_min"].append(time_minutes)
    results_lumped["T_wall_K"].append(T_wall_K)
    results_lumped["T_wall_F"].append(K_to_F(T_wall_K))
    results_lumped["h_in"].append(h_in)
    results_lumped["Q_in_sensible_kW"].append(Q_in_sensible / 1000.0)
    results_lumped["Q_in_post_comb_kW"].append(Q_post_comb_W / 1000.0) # FIX: Append the constant value
    results_lumped["h_out"].append(h_out)
    results_lumped["Q_out_kW"].append(Q_out / 1000.0)
    results_lumped["dTdt_K_min"].append(dTdt_K_sec * 60.0)

    T_wall_new = T_wall_K + dTdt_K_sec * dt_seconds

    # --- Check Targets ---
    dTdt_K_min_approx = dTdt_K_sec * 60.0
    if T_wall_K < params['T_wall_target1_K'] <= T_wall_new and time_target1_min is None:
        time_target1_min = time_minutes + (params['T_wall_target1_K'] - T_wall_K) / dTdt_K_min_approx if dTdt_K_min_approx > 0 else time_minutes
        print(f"--- Target 1 ({K_to_F(params['T_wall_target1_K']):.0f}°F) reached at ~{time_target1_min:.2f} min ---")
    if T_wall_K < params['T_wall_target2_K'] <= T_wall_new and time_target2_min is None:
        time_target2_min = time_minutes + (params['T_wall_target2_K'] - T_wall_K) / dTdt_K_min_approx if dTdt_K_min_approx > 0 else time_minutes
        print(f"--- Target 2 ({K_to_F(params['T_wall_target2_K']):.0f}°F) reached at ~{time_target2_min:.2f} min ---")

    T_wall_K = T_wall_new
    t_seconds += dt_seconds

    if int(t_seconds) % (5 * 60) == 0:
         print(f"Time: {t_seconds/60.0:.1f} min, Temp: {K_to_F(T_wall_K):.1f} °F")

# Add final state if loop finished normally or stopped by time limit
# Correct edge case: If stopped exactly at 240 min, t_seconds would be 240*60
time_minutes = min(t_seconds / 60.0, 240.0)
if T_wall_K > params['T_wall_target2_K']: T_wall_K = params['T_wall_target2_K'] # Cap at final target

# Add the final row only if it's distinct from the last recorded one
if len(results_lumped["Time_min"]) == 0 or results_lumped["Time_min"][-1] < time_minutes:
    results_lumped["Time_min"].append(time_minutes)
    results_lumped["T_wall_K"].append(T_wall_K)
    results_lumped["T_wall_F"].append(K_to_F(T_wall_K))
    # Recalc final state details
    h_in_final = calculate_h_in_py(params['T_gas_original_K'], T_wall_K, params['Re'], params['Pr'], params['gas_k_W_mK'], params['D_inner_mean_m'], params['epsilon_gas_est'], params['epsilon_wall_inner'], params['sigma'])
    h_out_final = calculate_h_out_py(T_wall_K, params['T_ambient_K'], params['epsilon_wall_outer'], params['sigma'], params['H_m'])
    Q_in_sensible_final = h_in_final * params['A_inner_total_m2'] * (params['T_gas_original_K'] - T_wall_K)
    Q_out_final = h_out_final * params['A_outer_total_m2'] * (T_wall_K - params['T_ambient_K'])
    try:
      if params['Mass_refractory_kg'] <= 0 or params['Cp_refractory_J_kgK'] <= 0: raise ValueError()
      dTdt_K_sec_final = (Q_in_sensible_final + Q_post_comb_W - max(0.0, Q_out_final)) / (params['Mass_refractory_kg'] * params['Cp_refractory_J_kgK'])
    except ValueError: dTdt_K_sec_final = 0
    results_lumped["h_in"].append(h_in_final)
    results_lumped["Q_in_sensible_kW"].append(Q_in_sensible_final / 1000.0)
    results_lumped["Q_in_post_comb_kW"].append(Q_post_comb_W / 1000.0)
    results_lumped["h_out"].append(h_out_final)
    results_lumped["Q_out_kW"].append(max(0.0, Q_out_final) / 1000.0)
    results_lumped["dTdt_K_min"].append(dTdt_K_sec_final * 60.0)

end_sim_time = time.time()
print(f"\nLumped Simulation (Post-Combustion, Corrected) complete.")
print(f"Python calculation took: {end_sim_time - start_sim_time:.2f} seconds.")
print(f"\n--- RESULTS (Lumped Model) ---")
if time_target1_min is not None:
    print(f"Time to reach {K_to_F(params['T_wall_target1_K']):.0f}°F (Target 1): {time_target1_min:.2f} minutes")
else:
    print(f"Target 1 ({K_to_F(params['T_wall_target1_K']):.0f}°F) was NOT reached within the simulation time ({time_minutes:.1f} min). Max Temp: {K_to_F(T_wall_K):.1f}°F")
if time_target2_min is not None:
    print(f"Time to reach {K_to_F(params['T_wall_target2_K']):.0f}°F (Target 2): {time_target2_min:.2f} minutes")
else:
    print(f"Target 2 ({K_to_F(params['T_wall_target2_K']):.0f}°F) was NOT reached within the simulation time ({time_minutes:.1f} min). Max Temp: {K_to_F(T_wall_K):.1f}°F")

# --- Create DataFrame, Save CSV, Plot ---
try:
    df_results_lumped = pd.DataFrame(results_lumped)
    csv_filename_lumped = "ladle_heating_sim_Lumped_PostComb_Corrected.csv"
    df_results_lumped.to_csv(csv_filename_lumped, index=False, float_format='%.3f')
    print(f"\nLumped (Post-Combustion, Corrected) results saved to {csv_filename_lumped}")

    plt.figure(figsize=(10, 6))
    plt.plot(df_results_lumped["Time_min"], df_results_lumped["T_wall_F"], label="Wall Temp (°F)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Temperature (°F)")
    plt.title("Lumped Model with Post-Combustion Heat (Corrected)")
    plt.axhline(y=K_to_F(params['T_wall_target1_K']), color='grey', linestyle='--', label=f'Target 1 ({K_to_F(params["T_wall_target1_K"]):.0f}°F)')
    plt.axhline(y=K_to_F(params['T_wall_target2_K']), color='r', linestyle='--', label=f'Target 2 ({K_to_F(params["T_wall_target2_K"]):.0f}°F)')
    if time_target1_min is not None: plt.axvline(x=time_target1_min, color='grey', linestyle=':', label=f'T1 Time ({time_target1_min:.1f} min)')
    if time_target2_min is not None: plt.axvline(x=time_target2_min, color='r', linestyle=':', label=f'T2 Time ({time_target2_min:.1f} min)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=K_to_F(params['T_wall_start_K']) - 100)
    plt.xlim(left=0)

    plot_filename_lumped = "ladle_heating_plot_Lumped_PostComb_Corrected.png"
    plt.savefig(plot_filename_lumped)
    print(f"Lumped (Post-Combustion, Corrected) plot saved to {plot_filename_lumped}")
    plt.show()

except ValueError as e:
    print(f"\nError creating DataFrame or plotting: {e}")
    print("Please check the lengths of the data lists in 'results_lumped':")
    for key, value in results_lumped.items():
        print(f"  {key}: {len(value)}")