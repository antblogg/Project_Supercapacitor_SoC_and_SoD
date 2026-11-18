import numpy as np
from matplotlib import pyplot
from pathlib import Path

from KalmanFilter import LinearKalmanFilter
from csvInterface import getPowerProfile, getVoltageProfile
from parameters import Parameters
from circuit.Circuit import Circuit

DATASETS = (
    ("assets/30.10.2026/fullChargeDischarge_processed_full.csv", "F"),
    ("assets/02.10.2026/usn-supercap_green-profile_no-recharge_partial_2026-09-25_processed_full.csv", "D"),
    ("assets/09.10.2026/2026-10-09_binabdu_run_1_processed_full.csv", "C"),
    ("assets/30.10.2026/power_profile_20kw_processed_full.csv", "E"),
    ("assets/09.10.2026/2026-10-09_binabdu_v2_processed_full.csv", "A"),
    ("assets/09.10.2026/2026-10-09_fredrik_2_processed_full.csv", "B"),
)

if not hasattr(Parameters, "V_MAX"):
    Parameters.V_MAX = 400


def main():
    output_dir = Path(__file__).parent / "output" / "SOCsim"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath, dataset_id in DATASETS:
        Parameters.DATA_FILEPATH = filepath

        power = getPowerProfile(filepath)
        voltage_reference = getVoltageProfile(filepath)
        result = LinearKalmanFilter().run(
            power,
            voltage_reference,
            state_bounds=(Parameters.V_MIN, Parameters.V_MAX),
        )

        time_seconds = result["time"]
        circuit = Circuit()
        soc_results = []
        for i in range(len(result["x"])):
            circuit.V = result["V_filt"][i]
            for j, branch in enumerate(circuit.circuitBranches):
                branch.V = result["x"][i][j]
            soc_results.append(circuit.calculateSocPercent() * 100)
        soc_estimates = np.asarray(soc_results)

        circuit = Circuit(InititalVoltage=voltage_reference[0])
        sim_volt = np.zeros(len(power))
        soc_results = []
        for i in range(len(power)):
            sim_volt[i] = circuit.computeNextState(power[i])
            soc_results.append(circuit.calculateSocPercent() * 100)

        soc_estimates_ol = np.asarray(soc_results)

        fig, (voltage_ax, soc_ax) = pyplot.subplots(
            2,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(8, 6),
        )
        fig.suptitle(f"Simulated Voltage and SoC Estimates - Dataset {dataset_id}")

        voltage_ax.set_title("Voltage")
        voltage_ax.set_ylabel("Voltage [V]")
        voltage_ax.plot(time_seconds, voltage_reference, label="Reference voltage")
        voltage_ax.plot(time_seconds, result["V_filt"], label="Kalman-filtered Voltage Estimate")
        voltage_ax.plot(time_seconds, sim_volt, label="Open-Loop Simulated Voltage")
        voltage_ax.legend(loc="lower left")
        voltage_ax.grid()

        soc_ax.set_title("State of Charge")
        soc_ax.set_xlabel("Time [s]")
        soc_ax.set_ylabel("SOC [%]")
        soc_ax.plot(time_seconds, result["V_filt"] / 4, label="Scaled voltage")
        soc_ax.plot(time_seconds, soc_estimates, label="Kalman SoC Estimate")
        soc_ax.plot(time_seconds, soc_estimates_ol, label="Open-Loop SoC Estimate")
        soc_ax.legend(loc="lower left")
        soc_ax.grid()

        fig.savefig(output_dir / f"{dataset_id}.png")
        pyplot.close(fig)


if __name__ == "__main__":
    main()
