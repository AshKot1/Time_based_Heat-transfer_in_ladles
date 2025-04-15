import numpy as np

def calculate_post_combustion_heat():
    """
    Calculates the heat released per second from post-combustion of CH4 and H2
    based on inlet/outlet concentrations and estimated flow rate.
    """
    print("--- Post-Combustion Heat Calculation ---")

    # --- Input Parameters (Based on User Provided Data) ---
    inlet_props = {
        "rho_kgm3": 0.59,                # USER INPUT - Change according to requirements (Gas Density)
        "v_mps": 95,                     # USER INPUT - Change according to requirements (Gas Velocity at Inlet)
        "M_mean_kg_mol": 0.02685,        # USER INPUT - Change according to requirements (Mean Molecular Weight, kg/mol)
        "x_CH4_in": 0.0214,              # USER INPUT - Change according to requirements (Inlet Mole Fraction CH4)
        "x_O2_in": 0.084,                # USER INPUT - Change according to requirements (Inlet Mole Fraction O2)
        "x_H2_in": 0.02905,              # USER INPUT - Change according to requirements (Inlet Mole Fraction H2)
    }
    outlet_props = {
        "x_CH4_out": 3.19e-07,           # USER INPUT - Change according to requirements (Outlet Mole Fraction CH4)
        "x_O2_out": 0.0188088,           # USER INPUT - Change according to requirements (Outlet Mole Fraction O2)
        "x_H2_out": 0.001064172,         # USER INPUT - Change according to requirements (Outlet Mole Fraction H2)
    }
    combustion_enthalpies = {
        # Note: Standard enthalpies of combustion (negative -> exothermic)
        # We use positive values here as we calculate heat *released*.
        "CH4_kJ_mol": 802.3,             # Physical Constant (Approx. LHV for reaction to H2O(g))
        "H2_kJ_mol": 241.8,              # Physical Constant (Approx. LHV for reaction to H2O(g))
    }

    # --- ASSUMPTION: Inlet Flow Area ---
    # Critical assumption affecting the flow rate and thus heat release.
    inlet_diameter_m = 0.13               # ASSUMPTION - Change according to requirements (Effective diameter where velocity 'v_mps' applies)
    inlet_area_m2 = np.pi * (inlet_diameter_m / 2)**2
    print(f"*** ASSUMPTION: Inlet Flow Area based on Diameter = {inlet_diameter_m:.2f} m -> Area = {inlet_area_m2:.3f} m² ***")

    # --- Calculations ---
    # 1. Mass Flow Rate (m_dot)
    m_dot_kg_s = inlet_props["rho_kgm3"] * inlet_props["v_mps"] * inlet_area_m2
    #m_dot_kg_s = 1.317824               # USER INPUT - Change according to requirements (The Mass flow rate exiting the burner outlet, ,received from CFD)
    print(f"Calculated Mass Flow Rate (m_dot): {m_dot_kg_s:.3f} kg/s")

    # 2. Total Molar Flow Rate (n_dot_total)
    # Ensure mean molar weight is positive
    if inlet_props["M_mean_kg_mol"] <= 0:
      print("Error: Mean molecular weight must be positive.")
      return 0.0
    n_dot_total_mol_s = m_dot_kg_s / inlet_props["M_mean_kg_mol"]
    print(f"Calculated Total Molar Flow Rate (n_dot_total): {n_dot_total_mol_s:.2f} mol/s")

    # 3. Moles Reacted per second
    delta_x_CH4 = inlet_props["x_CH4_in"] - outlet_props["x_CH4_out"]
    delta_x_H2 = inlet_props["x_H2_in"] - outlet_props["x_H2_out"]
    delta_x_O2 = inlet_props["x_O2_in"] - outlet_props["x_O2_out"]

    # Ensure deltas are non-negative (can't produce reactant)
    delta_x_CH4 = max(0.0, delta_x_CH4)
    delta_x_H2 = max(0.0, delta_x_H2)
    delta_x_O2 = max(0.0, delta_x_O2) # O2 consumed -> delta should be positive

    n_dot_CH4_reacted = n_dot_total_mol_s * delta_x_CH4
    n_dot_H2_reacted = n_dot_total_mol_s * delta_x_H2
    print(f"CH4 reacted: {n_dot_CH4_reacted:.3f} mol/s (based on delta_x={delta_x_CH4:.4f})")
    print(f"H2 reacted: {n_dot_H2_reacted:.3f} mol/s (based on delta_x={delta_x_H2:.4f})")

    # 4. Oxygen Consumption Check
    O2_needed_for_CH4 = n_dot_CH4_reacted * 2.0
    O2_needed_for_H2 = n_dot_H2_reacted / 2.0
    total_O2_needed = O2_needed_for_CH4 + O2_needed_for_H2
    n_dot_O2_reacted_calc = n_dot_total_mol_s * delta_x_O2
    print(f"Calculated O2 consumed: {n_dot_O2_reacted_calc:.3f} mol/s (based on delta_x={delta_x_O2:.4f})")
    print(f"Stoichiometric O2 needed for reacted CH4/H2: {total_O2_needed:.3f} mol/s")
    # Add a check for significant discrepancy
    if n_dot_O2_reacted_calc > 0 and abs(n_dot_O2_reacted_calc - total_O2_needed) / n_dot_O2_reacted_calc > 0.1: # Allow 10% difference
        print("Warning: Calculated O2 consumption differs significantly from stoichiometric requirement. Check input consistency.")

    # 5. Heat Release Calculation
    Q_dot_CH4_kJ_s = n_dot_CH4_reacted * combustion_enthalpies["CH4_kJ_mol"]
    Q_dot_H2_kJ_s = n_dot_H2_reacted * combustion_enthalpies["H2_kJ_mol"]
    Q_dot_post_combustion_kJ_s = Q_dot_CH4_kJ_s + Q_dot_H2_kJ_s
    Q_dot_post_combustion_kW = Q_dot_post_combustion_kJ_s
    Q_dot_post_combustion_W = Q_dot_post_combustion_kW * 1000
    Q_dot_post_combustion_MW = Q_dot_post_combustion_kJ_s / 1000

    print(f"\nHeat released from CH4 combustion: {Q_dot_CH4_kJ_s:.2f} kJ/s ({Q_dot_CH4_kJ_s:.2f} kW)")
    print(f"Heat released from H2 combustion: {Q_dot_H2_kJ_s:.2f} kJ/s ({Q_dot_H2_kJ_s:.2f} kW)")
    print(f"Total Post-Combustion Heat Release: {Q_dot_post_combustion_kW:.2f} kW")
    print(f"Total Post-Combustion Heat Release: {Q_dot_post_combustion_MW:.2f} MW")

    return Q_dot_post_combustion_W

# --- Execute calculation ---
Q_post_comb_W = calculate_post_combustion_heat()
#print(f"\nPower Generated by post combustion = {Q_post_comb_W:.3e} W")
print(f"\nPower Generated by post combustion = {Q_post_comb_W:.1f} W")

# Store for use in other scripts
_Q_post_comb_W = Q_post_comb_W