import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- Helper Functions (Define BEFORE use) ---
def K_to_F(T_K):
    if isinstance(T_K, float) and np.isnan(T_K): return np.nan
    try: return (T_K - 273.15) * 9/5 + 32
    except TypeError: return np.nan

# h_in Calculation - USES T_gas_original_K
def calculate_h_in_py(T_gas_K_original, T_wall_inner_K, Re, Pr, gas_k, D_mean, epsilon_gas, epsilon_wall_inner, sigma):
    T_wall_inner_avg = np.mean(T_wall_inner_K) if isinstance(T_wall_inner_K, np.ndarray) else T_wall_inner_K
    if T_wall_inner_avg is None or np.isnan(T_wall_inner_avg): T_wall_inner_avg = 0
    # Convection
    if Pr >= 0.6 and Re > 10000: Nu = 0.023 * (Re**0.8) * (Pr**0.4); h_conv = Nu * gas_k / D_mean
    else: h_conv = 50.0
    # Radiation
    h_rad = 0.0; deltaT = T_gas_K_original - T_wall_inner_avg
    if abs(deltaT) > 0.01:
        try:
            if T_gas_K_original > 0 and T_wall_inner_avg > 0: h_rad = sigma * epsilon_gas * epsilon_wall_inner * (T_gas_K_original**4 - T_wall_inner_avg**4) / deltaT
            else: h_rad = 0.0
        except (OverflowError, ZeroDivisionError): h_rad = 4*sigma*epsilon_gas*epsilon_wall_inner*T_gas_K_original**3 if T_gas_K_original > 0 else 0.0
    else: h_rad = 4*sigma*epsilon_gas*epsilon_wall_inner*T_gas_K_original**3 if T_gas_K_original > 0 else 0.0
    return h_conv + max(0.0, h_rad)

# h_out Calculation - Original
def calculate_h_out_py(T_wall_outer_K, T_ambient_K, epsilon_outer, sigma, L_char):
    T_wall_outer_avg = np.mean(T_wall_outer_K) if isinstance(T_wall_outer_K, np.ndarray) else T_wall_outer_K
    if T_wall_outer_avg is None or np.isnan(T_wall_outer_avg): T_wall_outer_avg = T_ambient_K
    # Natural Convection
    h_conv_nat = 2.0; deltaT = T_wall_outer_avg - T_ambient_K
    if deltaT > 0 and L_char > 0:
        try: h_conv_calc = 1.42 * (deltaT / L_char)**0.25; h_conv_nat = min(20.0, max(2.0, h_conv_calc))
        except (ValueError, OverflowError): h_conv_nat = 2.0
    # Radiation
    h_rad_out = 0.0; deltaT_rad = T_wall_outer_avg - T_ambient_K
    if abs(deltaT_rad) > 0.01:
        try:
            if T_wall_outer_avg > 0 and T_ambient_K > 0: h_rad_out = sigma * epsilon_outer * (T_wall_outer_avg**4 - T_ambient_K**4) / deltaT_rad
            else: h_rad_out = 0.0
        except (OverflowError, ZeroDivisionError): h_rad_out = 4*sigma*epsilon_outer*T_ambient_K**3 if T_ambient_K > 0 else 0.0
    else: h_rad_out = 4*sigma*epsilon_outer*T_ambient_K**3 if T_ambient_K > 0 else 0.0
    return h_conv_nat + max(0.0, h_rad_out)

# --- Parameters ---
params = {
  # --- Temperatures ---
  "T_wall_start_K": 1005.3722222222221,       # USER INPUT - Change according to requirements (Initial Wall Temp K)
  "T_wall_target1_K": 1372.0388888888888,     # USER INPUT - Change according to requirements (~2010 F - Inner Surface)
  "T_wall_target2_K": 1422.0388888888888,     # USER INPUT - Change according to requirements (~2100 F - Inner Surface)
  "T_gas_original_K": 1258.8249999999998,     # USER INPUT - Change according to requirements (~1806 F, used for sensible heat)
  "T_ambient_K": 298.15,                      # USER INPUT - Change according to requirements (Ambient Temp K)
  # --- Geometry & Mass (Derived) ---
  "A_inner_total_m2": 34.98648712421236,      # Derived (Needed for flux)
  "A_outer_total_m2": 42.95377026546754,      # Derived (Ref only)
  "Mass_refractory_kg": 26094.113670727387,    # Derived (Ref only)
  "thickness_m": 0.2286,                     # Derived
  "D_inner_mean_m": 2.2885400000000002,     # Derived
  "H_m": 4.39674,                           # Derived (Height)
  # --- Material Properties ---
  "Cp_refractory_J_kgK": 910.0,              # USER INPUT - Change according to requirements
  "k_refractory_W_mK": 1.75,               # USER INPUT - Change according to requirements
  "rho_refractory_kgm3": 2872.0,            # USER INPUT - Change according to requirements
  "epsilon_wall_inner": 0.8,               # ASSUMPTION - Change according to requirements
  "epsilon_wall_outer": 0.7,               # ASSUMPTION - Change according to requirements
  # --- Gas Properties (User provided) ---
  "gas_k_W_mK": 0.0454,                   # USER INPUT - Change according to requirements
  "Pr": 0.5369252775330396,                # USER INPUT - Change according to requirements
  "Re": 5058684.615348837,                 # USER INPUT - Change according to requirements
  "epsilon_gas_est": 0.15,                # ASSUMPTION - Change according to requirements
  # --- Constants ---
  "sigma": 5.67e-08
}
params['alpha_refractory'] = params['k_refractory_W_mK'] / (params['rho_refractory_kgm3'] * params['Cp_refractory_J_kgK']) if (params['rho_refractory_kgm3'] * params['Cp_refractory_J_kgK']) != 0 else 0

r_inner_avg = params['D_inner_mean_m'] / 2.0
r_outer_avg = r_inner_avg + params['thickness_m']

# --- Post-Combustion Heat Input ---
try: Q_post_comb_W = _Q_post_comb_W if '_Q_post_comb_W' in locals() else 663221.1
except NameError: Q_post_comb_W = 663221.1 # Default
q_dot_post_combustion_Wm2 = 0
if params['A_inner_total_m2'] > 0: q_dot_post_combustion_Wm2 = Q_post_comb_W / params['A_inner_total_m2']
print(f"--- Using Post-Combustion Heat Flux: {q_dot_post_combustion_Wm2:.1f} W/m² (Requires Inlet Area Assumption) ---")
print(f"--- Using T_gas_original: {params['T_gas_original_K']:.1f} K for sensible heat calc ---")
print(f"--- Using Original Heat Loss Calculation ---")

# --- Finite Difference Setup ---
N = 21 # USER INPUT - Change spatial resolution if needed
dr = params['thickness_m'] / (N - 1) if N > 1 else params['thickness_m']
r_nodes = np.linspace(r_inner_avg, r_outer_avg, N) if N > 1 else np.array([r_inner_avg])

# --- Stability & Time Step ---
courant_factor = 0.4
max_dt_stability = float('inf')
if N > 1 and dr > 0 and r_inner_avg * dr != 0 and params['alpha_refractory'] != 0:
    denominator = params['alpha_refractory'] * (1/dr**2 + 1/(r_inner_avg * dr))
    if denominator != 0: max_dt_stability = courant_factor / denominator
dt = min(max_dt_stability, 10.0) # ASSUMPTION - Limit max dt for stability/accuracy
dt = max(dt, 0.01) if dt > 0 else 0.01

print(f"\n--- 1D FDM Simulation Setup (Post-Combustion, Corrected) ---")
print(f"Nodes (N): {N}, dr: {dr:.4f} m")
print(f"Alpha: {params['alpha_refractory']:.3e} m²/s")
print(f"Max stable dt: {max_dt_stability:.3f} s, Chosen dt: {dt:.3f} s")

# --- Initialization ---
T = np.full(N, params['T_wall_start_K'])
T_new = np.copy(T)
t_seconds = 0.0
results_1d = {"Time_min": [], "T_inner_K": [], "T_outer_K": [], "T_inner_F": [], "T_outer_F": [], "h_in": [], "q_post_comb_kWm2": [], "h_out": []}
time_target1_min = None
time_target2_min = None
output_interval_seconds = 60.0
next_output_time = 0.0

print("\nStarting 1D simulation (Post-Combustion, Corrected)...")
start_sim_time = time.time()
max_time_simulate_hrs = 4 # ASSUMPTION - Safety limit on simulation time
max_t_seconds = max_time_simulate_hrs * 3600

# --- Simulation Loop ---
while T[0] < params['T_wall_target2_K']: # Loop until higher target (inner surface)
    if t_seconds > max_t_seconds: print(f"Warning: 1D Sim stopped after {max_time_simulate_hrs} hours."); break
    if dt <= 0: break # Safety check

    # Store results at intervals
    if t_seconds >= next_output_time:
        h_in_current = calculate_h_in_py(params['T_gas_original_K'], T[0], params['Re'], params['Pr'], params['gas_k_W_mK'], params['D_inner_mean_m'], params['epsilon_gas_est'], params['epsilon_wall_inner'], params['sigma'])
        h_out_current = calculate_h_out_py(T[N-1], params['T_ambient_K'], params['epsilon_wall_outer'], params['sigma'], params['H_m'])
        results_1d["Time_min"].append(t_seconds / 60.0)
        results_1d["T_inner_K"].append(T[0])
        results_1d["T_outer_K"].append(T[N-1])
        results_1d["T_inner_F"].append(K_to_F(T[0]))
        results_1d["T_outer_F"].append(K_to_F(T[N-1]))
        results_1d["h_in"].append(h_in_current)
        results_1d["q_post_comb_kWm2"].append(q_dot_post_combustion_Wm2/1000.0)
        results_1d["h_out"].append(h_out_current)
        next_output_time += output_interval_seconds
        if int(t_seconds / 60.0) % 5 == 0 and (t_seconds - (int(t_seconds / 60.0)//5 * 5 * 60)) < dt:
           print(f"Time: {t_seconds/60.0:.1f} min, T_inner: {K_to_F(T[0]):.1f} °F, T_outer: {K_to_F(T[N-1]):.1f} °F")

    # --- FDM Update ---
    h_in = calculate_h_in_py(params['T_gas_original_K'], T[0], params['Re'], params['Pr'], params['gas_k_W_mK'], params['D_inner_mean_m'], params['epsilon_gas_est'], params['epsilon_wall_inner'], params['sigma'])
    h_out = calculate_h_out_py(T[N-1], params['T_ambient_K'], params['epsilon_wall_outer'], params['sigma'], params['H_m'])

    # Interior Nodes
    if N > 2:
       for i in range(1, N - 1):
           if r_nodes[i] == 0 or dr == 0: continue
           term1 = (T[i+1] - 2*T[i] + T[i-1]) / dr**2
           term2 = (1 / r_nodes[i]) * (T[i+1] - T[i-1]) / (2 * dr)
           T_new[i] = T[i] + params['alpha_refractory'] * dt * (term1 + term2)

    # Boundary Nodes (Adjusted for Post-Combustion Flux)
    Fo = 0; combined_flux_term_inner = 0; Bi_out = 0
    if dr > 0: Fo = params['alpha_refractory'] * dt / dr**2
    if params['k_refractory_W_mK'] != 0 and dr > 0:
       combined_flux_term_inner = (h_in * (params['T_gas_original_K'] - T[0]) + q_dot_post_combustion_Wm2) * dr / params['k_refractory_W_mK']
       Bi_out = h_out * dr / params['k_refractory_W_mK']

    if N > 1:
        T_new[0] = T[0] + 2*Fo*( T[1] - T[0] + combined_flux_term_inner )
        T_new[N-1] = T[N-1] + 2*Fo*( T[N-2] - T[N-1] - Bi_out * (T[N-1] - params['T_ambient_K']) )
    elif N == 1: # Lumped if N=1
        Q_in_sens = h_in * params['A_inner_total_m2'] * (params['T_gas_original_K'] - T[0])
        Q_pc = q_dot_post_combustion_Wm2 * params['A_inner_total_m2']
        Q_o = h_out * params['A_outer_total_m2'] * (T[0] - params['T_ambient_K'])
        try: dTdt_sec = (Q_in_sens + Q_pc - max(0.0, Q_o)) / (params['Mass_refractory_kg'] * params['Cp_refractory_J_kgK'])
        except ValueError: dTdt_sec=0
        T_new[0] = T[0] + dTdt_sec * dt

    # Check stability
    if np.isnan(T_new).any() or np.isinf(T_new).any(): print("Error: NaN/Inf detected. Simulation unstable."); break

    # --- Check Targets ---
    if N > 0 :
       current_T_inner = T[0]; next_T_inner = T_new[0]
       dTdt_K_min_approx = (next_T_inner - current_T_inner) / dt * 60.0 if dt > 0 else 0
       if current_T_inner < params['T_wall_target1_K'] <= next_T_inner and time_target1_min is None:
           time_target1_min = (t_seconds / 60.0) + (params['T_wall_target1_K'] - current_T_inner) / dTdt_K_min_approx if dTdt_K_min_approx > 0 else (t_seconds / 60.0)
           print(f"--- Target 1 ({K_to_F(params['T_wall_target1_K']):.0f}°F Inner) reached at ~{time_target1_min:.2f} min ---")
       if current_T_inner < params['T_wall_target2_K'] <= next_T_inner and time_target2_min is None:
           time_target2_min = (t_seconds / 60.0) + (params['T_wall_target2_K'] - current_T_inner) / dTdt_K_min_approx if dTdt_K_min_approx > 0 else (t_seconds / 60.0)
           print(f"--- Target 2 ({K_to_F(params['T_wall_target2_K']):.0f}°F Inner) reached at ~{time_target2_min:.2f} min ---")

    # Update T and Time
    T = np.copy(T_new)
    t_seconds += dt

# --- Final State Logging & Reporting ---
time_final = min(t_seconds / 60.0, max_time_simulate_hrs * 60)
if not (np.isnan(T).any() or np.isinf(T).any()) and N > 0:
   T_inner_final = T[0] if T[0] <= params['T_wall_target2_K'] + (0.1 * dt / 60) else params['T_wall_target2_K'] # Cap near target
   T_outer_final = T[N-1]
else: T_inner_final = T_outer_final = np.nan # If sim failed

h_in_final = calculate_h_in_py(params['T_gas_original_K'], T_inner_final, params['Re'], params['Pr'], params['gas_k_W_mK'], params['D_inner_mean_m'], params['epsilon_gas_est'], params['epsilon_wall_inner'], params['sigma'])
h_out_final = calculate_h_out_py(T_outer_final, params['T_ambient_K'], params['epsilon_wall_outer'], params['sigma'], params['H_m'])

# Append final row only if distinct time
if len(results_1d["Time_min"]) == 0 or results_1d["Time_min"][-1] < time_final:
   results_1d["Time_min"].append(time_final)
   results_1d["T_inner_K"].append(T_inner_final)
   results_1d["T_outer_K"].append(T_outer_final)
   results_1d["T_inner_F"].append(K_to_F(T_inner_final))
   results_1d["T_outer_F"].append(K_to_F(T_outer_final))
   results_1d["h_in"].append(h_in_final)
   results_1d["q_post_comb_kWm2"].append(q_dot_post_combustion_Wm2/1000.0)
   results_1d["h_out"].append(h_out_final)

end_sim_time = time.time()
print(f"\n1D Simulation (Post-Combustion, Corrected) complete.")
print(f"Python calculation took: {end_sim_time - start_sim_time:.2f} seconds.")
print(f"\n--- RESULTS (1D FDM Model) ---")
if time_target1_min is not None: print(f"Time to reach {K_to_F(params['T_wall_target1_K']):.0f}°F (Target 1 - Inner): {time_target1_min:.2f} minutes")
else: print(f"Target 1 ({K_to_F(params['T_wall_target1_K']):.0f}°F) was NOT reached. Max Inner: {K_to_F(T_inner_final):.1f}°F at {time_final:.1f} min")
if time_target2_min is not None: print(f"Time to reach {K_to_F(params['T_wall_target2_K']):.0f}°F (Target 2 - Inner): {time_target2_min:.2f} minutes")
else: print(f"Target 2 ({K_to_F(params['T_wall_target2_K']):.0f}°F) was NOT reached. Max Inner: {K_to_F(T_inner_final):.1f}°F at {time_final:.1f} min")

# --- Create DataFrame, Save, Plot ---
try:
    df_results_1d = pd.DataFrame(results_1d)
    csv_filename_1d = "ladle_heating_sim_1D_PostComb_Corrected.csv"
    df_results_1d.to_csv(csv_filename_1d, index=False, float_format='%.3f')
    print(f"\n1D (Post-Combustion, Corrected) results saved to {csv_filename_1d}")

    plt.figure(figsize=(10, 6))
    plt.plot(df_results_1d["Time_min"], df_results_1d["T_inner_F"], label="Inner Wall Temp (°F)")
    plt.plot(df_results_1d["Time_min"], df_results_1d["T_outer_F"], label="Outer Wall Temp (°F)")
    plt.xlabel("Time (minutes)"); plt.ylabel("Temperature (°F)")
    plt.title("1D FDM with Post-Combustion Heat (Corrected)")
    plt.axhline(y=K_to_F(params['T_wall_target1_K']), color='grey', linestyle='--', label=f'Target 1 ({K_to_F(params["T_wall_target1_K"]):.0f}°F)')
    plt.axhline(y=K_to_F(params['T_wall_target2_K']), color='r', linestyle='--', label=f'Target 2 ({K_to_F(params["T_wall_target2_K"]):.0f}°F)')
    if time_target1_min is not None: plt.axvline(x=time_target1_min, color='grey', linestyle=':', label=f'T1 Time ({time_target1_min:.1f} min)')
    if time_target2_min is not None: plt.axvline(x=time_target2_min, color='r', linestyle=':', label=f'T2 Time ({time_target2_min:.1f} min)')
    plt.legend(); plt.grid(True); plt.ylim(bottom=K_to_F(params['T_wall_start_K']) - 100); plt.xlim(left=0)

    plot_filename_1d = "ladle_heating_plot_1D_PostComb_Corrected.png"
    plt.savefig(plot_filename_1d)
    print(f"1D (Post-Combustion, Corrected) plot saved to {plot_filename_1d}")
    plt.show()

    # Plot final temp profile if stable
    if not (np.isnan(T).any() or np.isinf(T).any()) and N>0:
       plt.figure(figsize=(8, 6))
       plt.plot(r_nodes, K_to_F(T)); plt.xlabel("Radius (m)"); plt.ylabel("Final Temperature (°F)")
       plt.title(f"1D Final Temp Profile (t = {time_final:.1f} min)"); plt.grid(True); plt.show()
except ValueError as e:
    print(f"\nError creating DataFrame or plotting 1D results: {e}")
    for key, value in results_1d.items(): print(f"  {key}: {len(value)}")