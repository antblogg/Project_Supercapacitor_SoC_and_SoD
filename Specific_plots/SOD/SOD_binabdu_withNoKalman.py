import numpy as np
from warnings import warn

from KalmanFilter import LinearKalmanFilter
from circuit.Circuit import Circuit
from csvInterface import getPowerProfile, getVoltageProfile, getCurrentProfile, getDataColumn
from parameters import Parameters
from plot_main import Application

if not hasattr(Parameters, "I_MAX"):
    Parameters.I_MAX = 190
if not hasattr(Parameters, "V_MIN"):
    Parameters.V_MIN = 1e-5
if not hasattr(Parameters, "V_MAX"):
    Parameters.V_MAX = 400

Parameters.DATA_FILEPATH = "assets/09.10.2026/2026-10-09_binabdu_run_1_processed_full.csv"

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
            warn(
                "Circuit simulation stopped at a voltage limit; "
                "remaining samples are not simulated.",
                RuntimeWarning,
                stacklevel=2,
            )
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
    app = Application(plot=False)
    power = getPowerProfile()
    voltage_reference = getVoltageProfile()
    cutoff_seconds = 256.4
    cutoff_index = int(cutoff_seconds / Parameters.DT)
    total_length = len(power)

    power_closed_loop = power[:cutoff_index]
    voltage_closed_loop = voltage_reference[:cutoff_index]
    result = LinearKalmanFilter().run(power_closed_loop, voltage_closed_loop)

    closed_loop_voltage = np.full(total_length, np.nan)
    closed_loop_voltage[:cutoff_index] = result["V_filt"]
    initial_state = result["x"][-1]
    initial_voltage = result["V_filt"][-1]

    circ = Circuit(InititalVoltage = 113)
    P = getPowerProfile()[cutoff_index:]

    naive_simulated_voltages = []
    open_loop_voltages = []

    
    open_loop_currents = []
    for power_level in (5_000, 11_000, 20_000):
        naive_simulated_voltage = [np.nan]*(len(P))
        open_loop_power = np.full(total_length - cutoff_index, power_level)
        for i in range(len(open_loop_power)):
            if power_level/circ.V >= 190:
                print(power_level/circ.V)
                break
            naive_simulated_voltage[i] = circ.computeNextState(power_level)
        naive_simulated_voltages.append(pad_open_loop(np.array(naive_simulated_voltage),total_length,cutoff_index))
        circ.reset()

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
    max_currents = [
        np.max(np.abs(current[np.isfinite(current)]))
        for current in open_loop_currents
    ]
    for power_level, sod, max_current in zip((5, 10, 20), sod_values, max_currents):
        print(
            f"SOD at {power_level} kW: {sod}; "
            f"maximum simulated current: {max_current:.2f} A"
        )

    app.ax.set_title("Note: Actual load ~= 20 kW")
    app.ax.legend(["V_ref", "V_cl", "SOD @ 5 kW", "SOD @ 11 kW", "SOD @ 20 kW","simple SOD @ 5 kW", "simple SOD @ 11 kW", "simple SOD @ 20 kW", "clamp state" ])
    app.drawPlot([voltage_reference, closed_loop_voltage, *open_loop_voltages,*naive_simulated_voltages,np.array(getDataColumn("clamp_active"))*100])


if __name__ == "__main__":
    main()
