import numpy as np
from warnings import warn
from matplotlib import pyplot

from KalmanFilter import LinearKalmanFilter
from circuit.Circuit import Circuit
from csvInterface import getPowerProfile, getVoltageProfile, getCurrentProfile, getDataColumn
from parameters import Parameters
from plot_main import Application


FILEPATH = None # None: use parameters.py
FILEPATH = "assets/30.10.2026/power_profile_20kw_processed_full.csv"
FILEPATH = "assets/30.10.2026/fullChargeDischarge_processed_full.csv"
FILEPATH = "assets/09.10.2026/2026-10-09_fredrik_2_processed_full.csv"
FILEPATH = "assets/09.10.2026/2026-10-09_binabdu_v2_processed_full.csv"  
FILEPATH = "assets/02.10.2026/usn-supercap_green-profile_no-recharge_partial_2026-09-25_processed_full.csv"
FILEPATH = "assets/09.10.2026/2026-10-09_binabdu_run_1_processed_full.csv" 

DATASET_NAME = "Dataset D"
START_TIME_SECONDS = 463.2
KW_DRAW = 20

if not hasattr(Parameters, "I_MAX"):
    Parameters.I_MAX = 190
if not hasattr(Parameters, "V_MIN"):
    Parameters.V_MIN = 1e-5
if not hasattr(Parameters, "V_MAX"):
    Parameters.V_MAX = 400


def get_sod(voltage, start_index=None):
    if start_index is None:
        valid_indices = np.flatnonzero(np.isfinite(voltage))
        start_index = valid_indices[0]

    remaining = voltage[start_index:]
    end = np.flatnonzero((remaining < 0) | ~np.isfinite(remaining))

    return end[0] * Parameters.DT


def simulate_circuit(power_profile, initial_state, initial_voltage):
    current_limit = Parameters.I_MAX
    voltage_min = Parameters.V_MIN
    voltage_max = Parameters.V_MAX
    circuit = Circuit(Parameters.RC_ORDER, initial_voltage)
    state = np.asarray(initial_state, dtype=float).reshape(-1)
    if len(state) != len(circuit.circuitBranches):
        raise ValueError("Kalman state does not match the circuit branch count")

    for branch, voltage in zip(circuit.circuitBranches, state):
        branch.V = voltage

    voltage_profile = np.full(len(power_profile), np.nan)
    current_profile = np.full(len(power_profile), np.nan)
    for index, power in enumerate(power_profile):
        current = power / circuit.V if circuit.V != 0 else 0.0
        current_profile[index] = current
        if abs(current) > current_limit:
            warn(
                "Circuit simulation stopped at the current limit; "
                "remaining samples are not simulated.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        next_voltage = circuit.computeTerminalVoltage(current)
        if (
            not np.isfinite(next_voltage)
            or next_voltage < voltage_min
            or next_voltage > voltage_max
        ):
            """
            warn(
                "Circuit simulation stopped at a voltage limit; "
                "remaining samples are not simulated.",
                RuntimeWarning,
                stacklevel=2,
            )
            """
            break
        voltage_profile[index] = circuit.computeNextState(power)
    return voltage_profile, current_profile


def pad_open_loop(voltage_profile, total_length, start_index):
    padded = np.full(total_length, np.nan)
    padded[start_index:] = np.where(voltage_profile > 0, voltage_profile, np.nan)
    return padded


def pad_current_profile(current_profile, total_length, start_index):
    padded = np.full(total_length, np.nan)
    padded[start_index:] = current_profile
    return padded


def main():
    cutoff_seconds = START_TIME_SECONDS 
    if FILEPATH is not None:
        Parameters.DATA_FILEPATH = FILEPATH 
    power = getPowerProfile()
    voltage_reference = getVoltageProfile()
    current_reference = getCurrentProfile()
    clamp = getDataColumn("clamp_active")
    cutoff_index = int(cutoff_seconds / Parameters.DT)
    total_length = len(power)

    power_closed_loop = power[:cutoff_index]
    voltage_closed_loop = voltage_reference[:cutoff_index]
    current_closed_loop = current_reference[:cutoff_index]
    result = LinearKalmanFilter().run(power_closed_loop, voltage_closed_loop)

    closed_loop_voltage = np.full(total_length, np.nan)
    closed_loop_voltage[:cutoff_index] = result["V_filt"]
    closed_loop_current = np.full(total_length, np.nan)
    closed_loop_current[:cutoff_index] = np.asarray(power_closed_loop) / np.asarray(closed_loop_voltage[:cutoff_index])
    initial_state = result["x"][-1]
    initial_voltage = result["V_filt"][-1]

    open_loop_voltages = []
    open_loop_currents = []
    for power_level in (5_000, 10_000, 20_000):
        open_loop_power = np.full(total_length - cutoff_index, power_level)
        simulated_voltage, simulated_current = simulate_circuit(
            open_loop_power, initial_state, initial_voltage
        )
        open_loop_voltages.append(
            pad_open_loop(simulated_voltage, total_length, cutoff_index)
        )
        open_loop_currents.append(
            pad_current_profile(simulated_current, total_length, cutoff_index)
        )

    sod_values = [get_sod(voltage, cutoff_index) for voltage in open_loop_voltages]

    true_sod_total = cutoff_seconds
    for i in range(len(clamp))[cutoff_index:]:
        if clamp[i] == 1:
            true_sod_total = i * Parameters.DT
            break
    true_sod_label = f"True SoD: {true_sod_total - cutoff_seconds:.0f}s"

    max_currents = [
        np.max(np.abs(current[np.isfinite(current)]))
        for current in open_loop_currents
    ]
    for power_level, sod, max_current in zip((5, 10, 20), sod_values, max_currents):
        print(
            f"SOD at {power_level} kW: {round(sod,1)}; "
            f"maximum simulated current: {max_current:.2f} A"
        )


    final_voltage = []
    for voltage in open_loop_voltages:
        for index in range(cutoff_index, len(voltage)):
            if np.isnan(voltage[index]):
                final_voltage.append(voltage[index - 1])
                break
    fig, (voltage_ax, current_ax) = pyplot.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(8, 6))
    fig.suptitle(f"Dataset: {DATASET_NAME}\nActual power demand: {KW_DRAW}kW")
    time_seconds = np.arange(total_length) * Parameters.DT
    plot_colors = ("tab:blue", "tab:purple","tab:orange", "tab:green", "tab:red")
    power_levels = (5, 10, 20)
    sod_labels = [
        f"SOD @ {power_level}kW: {sod:.0f}s"
        for power_level, sod in zip(power_levels, sod_values)
    ]

    voltage_ax.set_title("Voltage")
    voltage_ax.set_ylabel("Voltage [V]")
    voltage_ax.set_xlabel("Time [s]")
    voltage_ax.grid()
    voltage_ax.tick_params(axis="x", labelbottom=True)
    voltage_series = [voltage_reference, closed_loop_voltage,*open_loop_voltages]
    voltage_labels = ["Reference voltage", "Closed-loop voltage", *sod_labels]
    for series, color, label in zip(voltage_series, plot_colors, voltage_labels):
        if color == "tab:purple":
            linestyle = "-."
        else:
            linestyle = "-"
        voltage_ax.plot(time_seconds, series, linestyle = linestyle,color=color, label=label)
    voltage_ax.scatter(
        cutoff_seconds,
        voltage_reference[cutoff_index],
        color=plot_colors[0],
        edgecolors="gray",
        zorder=10,
    )
    for sod, color, voltage in zip(sod_values, plot_colors[2:], final_voltage):
        voltage_ax.scatter(
            cutoff_seconds + sod,
            voltage,
            color=color,
            edgecolors="gray",
            zorder=10,
        )
    voltage_ax.autoscale(False)
    voltage_ax.axvline(cutoff_seconds, color="black", linestyle="--", linewidth=1.0)
    voltage_ax.axvline(
        true_sod_total,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=true_sod_label,
    )
    voltage_ax.legend()

    current_ax.set_title("Current")
    current_ax.set_ylabel("Current [A]")
    current_series = [current_reference, closed_loop_current, *open_loop_currents]
    current_labels = ["Reference current", "Closed-loop current", *sod_labels]
    current_ax.grid()
    for series, color, label in zip(current_series, plot_colors, current_labels):
        if color == "tab:purple":
            linestyle = "-."
        else:
            linestyle = "-"
        current_ax.plot(time_seconds, series, color=color, linestyle=linestyle, label=label)
    current_ax.scatter(
        cutoff_seconds,
        current_reference[cutoff_index],
        color=plot_colors[0],
        edgecolors="gray",
        zorder=10,
    )
    for sod, color, current in zip(sod_values, plot_colors[2:], open_loop_currents):
        sod_index = cutoff_index + int(sod / Parameters.DT)
        current_ax.scatter(
            cutoff_seconds + sod,
            current[sod_index],
            color=color,
            edgecolors="gray",
            zorder=10,
        )
    current_ax.axvline(cutoff_seconds, color="black", linestyle="--", linewidth=1.0)
    current_ax.axvline(
        true_sod_total,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=true_sod_label,
    )
    current_ax.axhline(
        Parameters.I_MAX,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="Current limit",
        zorder=1,
    )
    current_ax.set_xlabel("Time [s]")
    current_ax.legend(loc="lower left")

    #fig.tight_layout()
    pyplot.show()
    



if __name__ == "__main__":
    main()
