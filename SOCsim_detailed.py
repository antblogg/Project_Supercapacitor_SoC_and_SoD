import numpy as np
from matplotlib import pyplot

from KalmanFilter import LinearKalmanFilter
from csvInterface import getPowerProfile, getVoltageProfile
from parameters import Parameters
from circuit.Circuit import Circuit


FILEPATH = "assets/30.10.2026/fullChargeDischarge_processed_full.csv"
FILEPATH = "assets/02.10.2026/usn-supercap_green-profile_no-recharge_partial_2026-09-25_processed_full.csv"  # D
FILEPATH = "assets/09.10.2026/2026-10-09_fredrik_2_processed_full.csv"  # B
FILEPATH = "assets/09.10.2026/2026-10-09_binabdu_v2_processed_full.csv"  # A
FILEPATH = "assets/30.10.2026/power_profile_20kw_processed_full.csv"  # E
FILEPATH = "assets/09.10.2026/2026-10-09_binabdu_run_1_processed_full.csv"  # C

if not hasattr(Parameters, "V_MAX"):
    Parameters.V_MAX = 400


def main():
    if FILEPATH is not None:
        Parameters.DATA_FILEPATH = FILEPATH

    power = getPowerProfile()
    voltage_reference = getVoltageProfile()
    result = LinearKalmanFilter().run(
        power, voltage_reference, state_bounds=(0.0, 400.0)
    )

    states = result["x"]
    circuit = Circuit(InititalVoltage=voltage_reference[0])
    soc_estimates = np.empty(len(states))
    for i, state in enumerate(states):
        circuit.V = result["V_filt"][i]
        for branch, branch_voltage in zip(circuit.circuitBranches, state):
            branch.V = branch_voltage
        soc_estimates[i] = circuit.calculateSocPercent() * 100

    time_seconds = result["time"]
    fig, (voltage_ax, state_ax) = pyplot.subplots(
        2,
        1,
        sharex=True,
        constrained_layout=True,
        figsize=(8, 6),
    )
    fig.suptitle("Reference and Kalman Voltage, RC States and SoC")

    voltage_ax.set_title("Voltage")
    voltage_ax.set_ylabel("Voltage [V]")
    voltage_ax.plot(time_seconds, voltage_reference, label="Reference voltage")
    voltage_ax.plot(time_seconds, result["V_filt"], label="Kalman voltage")
    voltage_ax.legend(loc="lower left")
    voltage_ax.grid()

    state_ax.set_title("Kalman RC states and SoC")
    state_ax.set_xlabel("Time [s]")
    state_ax.set_ylabel("SOC [%]")
    state_ax.plot(time_seconds, soc_estimates, color="tab:blue", label="SOC")
    soc_ax = state_ax.twinx()
    soc_ax.set_ylabel("Capacitor voltage [V]")
    soc_ax.plot(time_seconds, states[:, 0], color="tab:red", linestyle = "--", label="VC1")
    if states.shape[1] > 1:
        soc_ax.plot(time_seconds, states[:, 1], color="tab:green", linestyle = "--", label="VC2")
    
    state_ax.grid()


    state_lines, state_labels = state_ax.get_legend_handles_labels()
    soc_lines, soc_labels = soc_ax.get_legend_handles_labels()
    state_ax.legend(state_lines + soc_lines, state_labels + soc_labels, loc="best")

    pyplot.show()


if __name__ == "__main__":
    main()
