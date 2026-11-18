import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot

from KalmanFilter import LinearKalmanFilter
from parameters import Parameters
from circuit.Circuit import Circuit


POWER_LEVELS = (5_000, 10_000, 20_000)
MAX_SOD_SAMPLES = 500

DATASETS = (
    ("assets/30.10.2026/fullChargeDischarge_processed_full.csv", "F"),
    ("assets/02.10.2026/usn-supercap_green-profile_no-recharge_partial_2026-09-25_processed_full.csv", "D"),
    ("assets/09.10.2026/2026-10-09_binabdu_run_1_processed_full.csv", "C"),
    ("assets/30.10.2026/power_profile_20kw_processed_full.csv", "E"),
    ("assets/09.10.2026/2026-10-09_binabdu_v2_processed_full.csv", "A"),
    ("assets/09.10.2026/2026-10-09_fredrik_2_processed_full.csv", "B"),
)


def load_power_and_voltage(filepath=None):
    """Load profiles using the column names supported by csvInterface."""
    data_path = Path(__file__).resolve().parent / (filepath or Parameters.DATA_FILEPATH)
    with data_path.open(newline="") as data_file:
        reader = csv.DictReader(data_file)
        columns = reader.fieldnames or []
        rows = list(reader)

    def read_column(names):
        for name in names:
            if name in columns:
                try:
                    return np.asarray([float(row[name]) for row in rows])
                except ValueError as error:
                    raise ValueError(f"Non-numeric data in column {name!r}") from error
        return None

    voltage = read_column(("sc_v", "V_sc_V", "grid_v", "V_grid_V"))
    if voltage is None:
        raise ValueError(f"No supported voltage column found in {data_path}")

    current = read_column(("sc_a_signed", "sc_a", "I_sc_A", "mean_a", "I_grid_A"))
    if current is None:
        power = read_column(("P_grid_W",))
        if power is None:
            raise ValueError(f"No supported current or power column found in {data_path}")
    else:
        power = current * voltage

    return power, voltage


def calculate_sod_profiles(states, voltage_profile):
    states = np.asarray(states, dtype=float)
    voltage_profile = np.asarray(voltage_profile, dtype=float)

    step = max(1, int(np.ceil(len(states) / MAX_SOD_SAMPLES)))
    indices = np.arange(0, len(states), step)

    sod_profiles = {}

    for power in POWER_LEVELS:
        sod = np.full(len(indices), np.nan)

        for out_idx, state_idx in enumerate(indices):
            state = states[state_idx]
            voltage = voltage_profile[state_idx]

            if not np.isfinite(voltage):
                continue

            circuit = Circuit(Parameters.RC_ORDER, voltage)
            circuit.V = voltage

            for branch, branch_voltage in zip(circuit.circuitBranches, state):
                branch.V = branch_voltage

            sod[out_idx] = circuit.calculateSod(power)

        sod_profiles[power] = sod

    return sod_profiles, indices


def main():
    output_dir = Path(__file__).parent / "output" / "SODprofile"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath, dataset_id in DATASETS:
        power, voltage_reference = load_power_and_voltage(filepath)
        result = LinearKalmanFilter().run(power, voltage_reference)

        sod_profiles, indices = calculate_sod_profiles(
            result["x"],
            result["V_filt"],
        )

        time_seconds = result["time"][indices]
        voltage_filtered = result["V_filt"][indices]

        fig, (voltage_ax, sod_ax) = pyplot.subplots(
            2,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(10, 7),
        )

        fig.suptitle(f"Kalman Filtered Voltage and SOD - Dataset {dataset_id}")

        voltage_ax.plot(time_seconds, voltage_filtered, color="tab:blue")
        voltage_ax.set_title("Kalman Filtered Voltage")
        voltage_ax.set_ylabel("Voltage [V]")
        voltage_ax.grid()

        for power_level, sod in sod_profiles.items():
            voltage_kw = power_level // 1_000
            sod_ax.plot(time_seconds, sod, label=f"{voltage_kw} kW")

        sod_ax.set_title("SoD")
        sod_ax.set_xlabel("Time [s]")
        sod_ax.set_ylabel("SOD [s]")
        sod_ax.grid()
        sod_ax.legend()

        fig.savefig(output_dir / f"{dataset_id}.png")
        pyplot.close(fig)


if __name__ == "__main__":
    main()
