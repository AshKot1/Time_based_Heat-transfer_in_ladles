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
  "T_wall_start_K": 1005.37,       # USER INPUT - Change according to requirements (Initial Wall Temp K)
  "T_wall_target1_K": 1372.04,     # USER INPUT - Change according to requirements (~2010 F - Avg Inner)
  "T_wall_target2_K": 1422.04,     # USER INPUT - Change according to requirements (~2100 F - Avg Inner)
  "T_gas_original_K": 1258.825,     # USER INPUT - Change according to requirements (~1806 F)
  "T_ambient_K": 298.15,                      # USER INPUT - Change according to requirements
  # --- Geometry & Mass (Derived) ---
  "A_inner_total_m2": 34.9865,      # Derived (Flux)
  "A_outer_total_m2": 42.9538,      # Derived (Ref)
  "Mass_refractory_kg": 26094.114,    # Derived (Ref)
  "thickness_m": 0.2286,                     # Derived
  "D_inner_mean_m": 2.2885,     # Derived (Also L_char_top)
  "H_m": 4.397,                           # Derived (Height, L_char_side)
  # --- Material Properties ---
  "Cp_refractory_J_kgK": 910.0,              # USER INPUT - Change according to requirements
  "k_refractory_W_mK": 1.75,               # USER INPUT - Change according to requirements
  "rho_refractory_kgm3": 2872.0,            # USER INPUT - Change according to requirements
  "epsilon_wall_inner": 0.8,               # ASSUMPTION - Change according to requirements
  "epsilon_wall_outer": 0.7,               # ASSUMPTION - Change according to requirements
  # --- Gas Properties (User provided) ---
  "gas_k_W_mK": 0.0454,                   # USER INPUT - Change according to requirements
  "Pr": 0.5369253,                # USER INPUT - Change according to requirements
  "Re": 5058684.61535,                 # USER INPUT - Change according to requirements
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

# --- 2D Finite Difference Setup ---
Nr = 11 # USER INPUT - Change spatial resolution if needed
Nz = 21 # USER INPUT - Change spatial resolution if needed
dr = params['thickness_m'] / (Nr - 1) if Nr > 1 else params['thickness_m']
dz = params['H_m'] / (Nz - 1) if Nz > 1 else params['H_m']
r_nodes = np.linspace(r_inner_avg, r_outer_avg, Nr) if Nr > 1 else np.array([r_inner_avg])
z_nodes = np.linspace(0, params['H_m'], Nz) if Nz > 1 else np.array([0])

# --- Stability & Time Step ---
courant_factor = 0.4
max_dt_stability_2d = float('inf')
if Nr>1 and Nz>1 and dr>0 and dz>0 and r_inner_avg*dr*params['alpha_refractory']!=0:
   denom_dt_check = params['alpha_refractory'] * ( (1/dr**2 + 1/(r_inner_avg * dr)) + 1/dz**2 )
   if denom_dt_check != 0: max_dt_stability_2d = courant_factor / denom_dt_check
dt = min(max_dt_stability_2d, 2.0) # ASSUMPTION - Limit max dt
dt = max(dt, 0.001) if dt > 0 else 0.001

print(f"\n--- 2D FDM Simulation Setup (Post-Combustion, Corrected) ---")
print(f"Nodes: Nr={Nr}, Nz={Nz}. Steps: dr={dr:.4f} m, dz={dz:.4f} m")
print(f"Alpha: {params['alpha_refractory']:.3e} m²/s")
print(f"Max stable dt: {max_dt_stability_2d:.3f} s, Chosen dt: {dt:.3f} s")

# --- Initialization ---
T = np.full((Nr, Nz), params['T_wall_start_K'])
T_new = np.copy(T)
# Pre-calc factors
Fo_r, Fo_z = 0, 0; r_inv = np.zeros_like(r_nodes); dr_inv, dz_inv = 0, 0
if dr > 0: Fo_r = params['alpha_refractory'] * dt / dr**2; dr_inv = 1.0 / dr
if dz > 0: Fo_z = params['alpha_refractory'] * dt / dz**2; dz_inv = 1.0 / dz
non_zero_r_mask = r_nodes != 0
if np.any(non_zero_r_mask): r_inv[non_zero_r_mask] = 1.0 / r_nodes[non_zero_r_mask]

t_seconds = 0.0
results_2d = {"Time_min": [], "T_inner_avg_K": [], "T_outer_avg_K": [], "T_inner_avg_F": [], "T_outer_avg_F": []}
time_target1_min = None; time_target2_min = None
output_interval_seconds = 60.0; next_output_time = 0.0

print("\nStarting 2D simulation (Post-Combustion, Corrected)...")
start_sim_time = time.time()
max_time_simulate_hrs = 4 # ASSUMPTION - Safety limit
max_t_seconds = max_time_simulate_hrs * 3600

# --- Simulation Loop ---
current_T_inner_avg = np.mean(T[0, :])
while current_T_inner_avg < params['T_wall_target2_K']:
    if t_seconds > max_t_seconds: print(f"Warning: 2D Sim stopped after {max_time_simulate_hrs} hours."); break
    if dt <= 0: break

    # Store results
    if t_seconds >= next_output_time:
        outer_temps_avg = np.mean(T[Nr-1, :]) if Nr > 0 else np.nan
        results_2d["Time_min"].append(t_seconds / 60.0)
        results_2d["T_inner_avg_K"].append(current_T_inner_avg)
        results_2d["T_outer_avg_K"].append(outer_temps_avg)
        results_2d["T_inner_avg_F"].append(K_to_F(current_T_inner_avg))
        results_2d["T_outer_avg_F"].append(K_to_F(outer_temps_avg))
        next_output_time += output_interval_seconds
        if int(t_seconds / 60.0) % 5 == 0 and (t_seconds - (int(t_seconds / 60.0)//5 * 5 * 60)) < dt:
             print(f"Time: {t_seconds/60.0:.1f} min, Avg T_inner: {K_to_F(current_T_inner_avg):.1f} °F, Avg T_outer: {K_to_F(outer_temps_avg):.1f} °F")

    # --- Update FDM ---
    h_in = calculate_h_in_py(params['T_gas_original_K'], T[0, :], params['Re'], params['Pr'], params['gas_k_W_mK'], params['D_inner_mean_m'], params['epsilon_gas_est'], params['epsilon_wall_inner'], params['sigma'])
    h_out = calculate_h_out_py(T[Nr-1, :], params['T_ambient_K'], params['epsilon_wall_outer'], params['sigma'], params['H_m'])
    h_top = calculate_h_out_py(T[:, Nz-1], params['T_ambient_K'], params['epsilon_wall_outer'], params['sigma'], params['D_inner_mean_m']) # Use D_mean approx

    Bi_out_dr, Bi_top_dz = 0, 0
    if params['k_refractory_W_mK'] != 0:
        if dr > 0 : Bi_out_dr = h_out * dr / params['k_refractory_W_mK']
        if dz > 0 : Bi_top_dz = h_top * dz / params['k_refractory_W_mK']

    # Interior Nodes
    if Nr > 2 and Nz > 2:
        for i in range(1, Nr - 1):
            for j in range(1, Nz - 1):
                dTdr_term = r_inv[i] * (T[i+1, j] - T[i-1, j]) * 0.5 * dr_inv if (r_nodes[i]!=0 and dr!=0) else 0
                d2Tdr2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) * dr_inv**2 if dr != 0 else 0
                d2Tdz2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) * dz_inv**2 if dz != 0 else 0
                T_new[i, j] = T[i, j] + params['alpha_refractory'] * dt * (d2Tdr2 + dTdr_term + d2Tdz2)

    # Boundaries (excluding corners)
    if Nz > 2: # Sides
        for j in range(1, Nz - 1):
            if Nr > 1: # Inner Boundary (Modified for Flux)
               local_flux_term_inner = (h_in * (params['T_gas_original_K'] - T[0,j]) + q_dot_post_combustion_Wm2) * dr / params['k_refractory_W_mK'] if (params['k_refractory_W_mK']!=0 and dr!=0) else 0
               d2Tdz2_in = (T[0, j+1] - 2*T[0, j] + T[0, j-1]) * dz_inv**2 if dz!=0 else 0
               radial_term_in = 2*Fo_r * ( T[1, j] - T[0, j] + local_flux_term_inner ) if dr!=0 else 0
               T_new[0, j] = T[0, j] + params['alpha_refractory'] * dt * d2Tdz2_in + radial_term_in
            if Nr > 1: # Outer Boundary
               d2Tdz2_out = (T[Nr-1, j+1] - 2*T[Nr-1, j] + T[Nr-1, j-1]) * dz_inv**2 if dz!=0 else 0
               radial_term_out = 2*Fo_r * ( T[Nr-2, j] - T[Nr-1, j] - Bi_out_dr * (T[Nr-1, j] - params['T_ambient_K']) ) if dr!=0 else 0
               T_new[Nr-1, j] = T[Nr-1, j] + params['alpha_refractory'] * dt * d2Tdz2_out + radial_term_out
    if Nr > 2: # Top/Bottom
       for i in range(1, Nr - 1):
           if Nz > 1: # Bottom (Adiabatic)
               d2Tdr2_bot = (T[i+1, 0] - 2*T[i, 0] + T[i-1, 0]) * dr_inv**2 if dr!=0 else 0
               dTdr_term_bot = r_inv[i] * (T[i+1, 0] - T[i-1, 0]) * 0.5 * dr_inv if (r_nodes[i]!=0 and dr!=0) else 0
               axial_term_bot = 2*Fo_z*(T[i, 1]-T[i, 0]) if dz!=0 else 0
               T_new[i, 0] = T[i, 0] + params['alpha_refractory'] * dt * (d2Tdr2_bot + dTdr_term_bot) + axial_term_bot
           if Nz > 1: # Top (Convection)
               d2Tdr2_top = (T[i+1, Nz-1] - 2*T[i, Nz-1] + T[i-1, Nz-1]) * dr_inv**2 if dr!=0 else 0
               dTdr_term_top = r_inv[i] * (T[i+1, Nz-1] - T[i-1, Nz-1]) * 0.5 * dr_inv if (r_nodes[i]!=0 and dr!=0) else 0
               axial_term_top_conv = 2*Fo_z*(T[i, Nz-2] - T[i, Nz-1] - Bi_top_dz*(T[i, Nz-1] - params['T_ambient_K'])) if dz!=0 else 0
               T_new[i, Nz-1] = T[i, Nz-1] + params['alpha_refractory']*dt*(d2Tdr2_top + dTdr_term_top) + axial_term_top_conv

    # Corner Nodes
    if Nr > 1 and Nz > 1:
       # Inner Bottom (0, 0)
       local_flux_00 = (h_in * (params['T_gas_original_K'] - T[0,0]) + q_dot_post_combustion_Wm2)*dr/params['k_refractory_W_mK'] if (params['k_refractory_W_mK']!=0 and dr!=0) else 0
       rad_term = 2*Fo_r * (T[1, 0] - T[0, 0] + local_flux_00) if dr!=0 else 0
       ax_term = 2*Fo_z * (T[0, 1] - T[0, 0]) if dz!=0 else 0
       T_new[0, 0] = T[0, 0] + rad_term + ax_term
       # Outer Bottom (Nr-1, 0)
       rad_term = 2*Fo_r * (T[Nr-2, 0] - T[Nr-1, 0] - Bi_out_dr * (T[Nr-1, 0] - params['T_ambient_K'])) if dr!=0 else 0
       ax_term = 2*Fo_z * (T[Nr-1, 1] - T[Nr-1, 0]) if dz!=0 else 0
       T_new[Nr-1, 0] = T[Nr-1, 0] + rad_term + ax_term
       # Inner Top (0, Nz-1)
       local_flux_0N = (h_in * (params['T_gas_original_K'] - T[0,Nz-1]) + q_dot_post_combustion_Wm2)*dr/params['k_refractory_W_mK'] if (params['k_refractory_W_mK']!=0 and dr!=0) else 0
       rad_term = 2*Fo_r * (T[1, Nz-1] - T[0, Nz-1] + local_flux_0N) if dr!=0 else 0
       ax_term = 2*Fo_z * (T[0, Nz-2] - T[0, Nz-1] - Bi_top_dz * (T[0, Nz-1] - params['T_ambient_K'])) if dz!=0 else 0
       T_new[0, Nz-1] = T[0, Nz-1] + rad_term + ax_term
       # Outer Top (Nr-1, Nz-1)
       rad_term = 2*Fo_r * (T[Nr-2, Nz-1] - T[Nr-1, Nz-1] - Bi_out_dr * (T[Nr-1, Nz-1] - params['T_ambient_K'])) if dr!=0 else 0
       ax_term = 2*Fo_z * (T[Nr-1, Nz-2] - T[Nr-1, Nz-1] - Bi_top_dz * (T[Nr-1, Nz-1] - params['T_ambient_K'])) if dz!=0 else 0
       T_new[Nr-1, Nz-1] = T[Nr-1, Nz-1] + rad_term + ax_term

    # Check stability
    if np.isnan(T_new).any() or np.isinf(T_new).any(): print("Error: NaN/Inf detected. Sim unstable."); break

    # --- Check Targets ---
    next_T_inner_avg = np.mean(T_new[0, :])
    dTdt_avg_K_min_approx = (next_T_inner_avg - current_T_inner_avg) / dt * 60.0 if dt > 0 else 0
    if current_T_inner_avg < params['T_wall_target1_K'] <= next_T_inner_avg and time_target1_min is None:
        time_target1_min = (t_seconds / 60.0) + (params['T_wall_target1_K'] - current_T_inner_avg) / dTdt_avg_K_min_approx if dTdt_avg_K_min_approx > 0 else (t_seconds / 60.0)
        print(f"--- Target 1 ({K_to_F(params['T_wall_target1_K']):.0f}°F Avg Inner) reached at ~{time_target1_min:.2f} min ---")
    if current_T_inner_avg < params['T_wall_target2_K'] <= next_T_inner_avg and time_target2_min is None:
        time_target2_min = (t_seconds / 60.0) + (params['T_wall_target2_K'] - current_T_inner_avg) / dTdt_avg_K_min_approx if dTdt_avg_K_min_approx > 0 else (t_seconds / 60.0)
        print(f"--- Target 2 ({K_to_F(params['T_wall_target2_K']):.0f}°F Avg Inner) reached at ~{time_target2_min:.2f} min ---")

    # Update T and Time
    T = np.copy(T_new)
    current_T_inner_avg = np.mean(T[0,:])
    t_seconds += dt

# --- Final Logging & Saving ---
time_final = min(t_seconds / 60.0, max_time_simulate_hrs*60)
if not (np.isnan(T).any() or np.isinf(T).any()) and Nr>0 and Nz>0:
    inner_temps_avg = np.mean(T[0, :])
    outer_temps_avg = np.mean(T[Nr-1, :])
    if inner_temps_avg > params['T_wall_target2_K']: inner_temps_avg = params['T_wall_target2_K'] # Cap
else: inner_temps_avg = outer_temps_avg = np.nan

# Add final row only if distinct
if len(results_2d["Time_min"]) == 0 or results_2d["Time_min"][-1] < time_final:
   results_2d["Time_min"].append(time_final)
   results_2d["T_inner_avg_K"].append(inner_temps_avg)
   results_2d["T_outer_avg_K"].append(outer_temps_avg)
   results_2d["T_inner_avg_F"].append(K_to_F(inner_temps_avg))
   results_2d["T_outer_avg_F"].append(K_to_F(outer_temps_avg))

end_sim_time = time.time()
print(f"\n2D Simulation (Post-Combustion, Corrected) complete.")
print(f"Python calculation took: {end_sim_time - start_sim_time:.2f} seconds.")
print(f"\n--- RESULTS (2D FDM Model) ---")
if time_target1_min is not None: print(f"Time to reach {K_to_F(params['T_wall_target1_K']):.0f}°F (Target 1 - Avg Inner): {time_target1_min:.2f} minutes")
else: print(f"Target 1 ({K_to_F(params['T_wall_target1_K']):.0f}°F) was NOT reached. Max Avg Inner: {K_to_F(inner_temps_avg):.1f}°F at {time_final:.1f} min")
if time_target2_min is not None: print(f"Time to reach {K_to_F(params['T_wall_target2_K']):.0f}°F (Target 2 - Avg Inner): {time_target2_min:.2f} minutes")
else: print(f"Target 2 ({K_to_F(params['T_wall_target2_K']):.0f}°F) was NOT reached. Max Avg Inner: {K_to_F(inner_temps_avg):.1f}°F at {time_final:.1f} min")


try:
    df_results_2d = pd.DataFrame(results_2d)
    csv_filename_2d = "ladle_heating_sim_2D_PostComb_Corrected.csv"
    df_results_2d.to_csv(csv_filename_2d, index=False, float_format='%.3f')
    print(f"\n2D (Post-Combustion, Corrected) results saved to {csv_filename_2d}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_results_2d["Time_min"], df_results_2d["T_inner_avg_F"], label="Avg Inner Wall Temp (°F)")
    plt.plot(df_results_2d["Time_min"], df_results_2d["T_outer_avg_F"], label="Avg Outer Wall Temp (°F)")
    plt.xlabel("Time (minutes)"); plt.ylabel("Avg Temperature (°F)")
    plt.title("2D Avg Temp vs. Time (FDM with Post-Combustion Heat, Corrected)")
    plt.axhline(y=K_to_F(params['T_wall_target1_K']), color='grey', linestyle='--', label=f'Target 1 ({K_to_F(params["T_wall_target1_K"]):.0f}°F)')
    plt.axhline(y=K_to_F(params['T_wall_target2_K']), color='r', linestyle='--', label=f'Target 2 ({K_to_F(params["T_wall_target2_K"]):.0f}°F)')
    if time_target1_min is not None: plt.axvline(x=time_target1_min, color='grey', linestyle=':', label=f'T1 Time ({time_target1_min:.1f} min)')
    if time_target2_min is not None: plt.axvline(x=time_target2_min, color='r', linestyle=':', label=f'T2 Time ({time_target2_min:.1f} min)')
    plt.legend(); plt.grid(True); plt.ylim(bottom=K_to_F(params['T_wall_start_K'])-100); plt.xlim(left=0)
    plot_filename_2d = "ladle_heating_plot_2D_avg_PostComb_Corrected.png"
    plt.savefig(plot_filename_2d)
    print(f"2D avg temp plot (Post-Combustion, Corrected) saved to {plot_filename_2d}")
    plt.show()

    # Plot final distribution if stable
    if not (np.isnan(T).any() or np.isinf(T).any()) and Nr>0 and Nz>0:
       plt.figure(figsize=(6, 10))
       R_grid, Z_grid = np.meshgrid(r_nodes, z_nodes, indexing='ij')
       temp_F_final = K_to_F(T); contour = plt.contourf(R_grid, Z_grid, temp_F_final, cmap='inferno', levels=25)
       plt.colorbar(contour, label='Temperature (°F)'); plt.xlabel("Radius (m)"); plt.ylabel("Height (m)")
       plt.title(f"2D Final Temp Dist (t = {time_final:.1f} min)"); plt.axis('equal')
       plot_filename_2d_contour = "ladle_heating_plot_2D_contour_PostComb_Corrected.png"
       plt.savefig(plot_filename_2d_contour)
       print(f"2D contour plot (Post-Combustion, Corrected) saved to {plot_filename_2d_contour}")
       plt.show()

except ValueError as e:
    print(f"\nError creating DataFrame or plotting 2D results: {e}")
    for key, value in results_2d.items(): print(f"  {key}: {len(value)}")
except Exception as e:
     print(f"\nAn unexpected error occurred during plotting/saving 2D results: {e}")
