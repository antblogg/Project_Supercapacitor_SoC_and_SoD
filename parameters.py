import numpy as np
class Parameters:
    DATA_FILEPATH = "assets/30.10.2026/fullChargeDischarge_processed_full.csv" #N/A
    DATA_FILEPATH = "assets/09.10.2026/2026-10-09_binabdu_v2_processed_full.csv"  # A
    DATA_FILEPATH = "assets/02.10.2026/usn-supercap_green-profile_no-recharge_partial_2026-09-25_processed_full.csv" # D
    DATA_FILEPATH = "assets/30.10.2026/power_profile_20kw_processed_full.csv" # E
    DATA_FILEPATH = "assets/09.10.2026/2026-10-09_fredrik_2_processed_full.csv" #B
    DATA_FILEPATH = "assets/09.10.2026/2026-10-09_binabdu_run_1_processed_full.csv" #C
    
    DT = 0.05  # simulation time step size
    RC_ORDER = 1

    V_MAX = 390 # maximum possible voltage 
    V_MIN = 1e-5 # minimum possible voltage 

    I_MAX = 190

    # list of resistor values
    R_L = 1000 # leakage
    R = [
        0.043391,
        889.625,
        693.38,
        340.121,
    ]
    C = [
        16.67,
        7.02254,
        8.79011,
        9.46376,
    ]
    # initial capacitor voltages      
    InitialTerminalVoltage = 250
    # TODO Set initial voltage based on dataset instead. Given as V_T(t=0)
    V0 = [InitialTerminalVoltage] * 5
    

    # a bunch of sanity checks on parameter values:
    assert DT > 0
    assert RC_ORDER >=0
    assert RC_ORDER <=10 
    assert type(RC_ORDER) == int
    for r in R:
        assert r > 0 
    for c in C:
        assert c > 0 

    assert InitialTerminalVoltage > 0
