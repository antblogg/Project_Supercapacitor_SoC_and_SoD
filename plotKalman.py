import numpy as np
from matplotlib import pyplot

from KalmanFilter import LinearKalmanFilter
from csvInterface import getPowerProfile, getVoltageProfile
from parameters import Parameters


paths = []
FILEPATH = "assets/02.10.2026/usn-supercap_green-profile_no-recharge_partial_2026-09-25_processed_full.csv"
paths.append(FILEPATH)


def low_pass(signal, alpha=0.05):
    filtered = np.copy(signal)
    valid = np.isfinite(signal)
    indices = np.where(valid)[0]

    if len(indices) == 0:
        return filtered

    filtered[indices[0]] = signal[indices[0]]
    for i in indices[1:]:
        prev = filtered[i - 1]
        if not np.isfinite(prev):
            prev = signal[i]
        filtered[i] = alpha * signal[i] + (1 - alpha) * prev

    return filtered


def main():
    for FILEPATH in paths:
        Parameters.DATA_FILEPATH = FILEPATH

        power = getPowerProfile()
        voltage_reference = getVoltageProfile()

        result = LinearKalmanFilter().run(power, voltage_reference, state_bounds=(Parameters.V_MIN, Parameters.V_MAX))

        time_seconds = result["time"]
        innovation_filtered = low_pass(result["innov"])

        fig, axes = pyplot.subplots(
            3,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(8, 6),
        )
        fig.suptitle("Kalman Filter Results - Dataset D")

        axes[0].set_title("Voltage Estimate")
        axes[0].set_ylabel("Voltage [V]")
        axes[0].plot(time_seconds, voltage_reference, label="Reference Voltage")
        axes[0].plot(time_seconds, result["V_filt"], label="Kalman Voltage")
        axes[0].legend(loc="lower left")
        axes[0].grid()

        axes[1].set_title("Estimated States")
        axes[1].set_ylabel("Branch Voltage [V]")
        axes[1].plot(time_seconds, result["x"][:, 0], label=r"$V_{C0}$")
        if result["x"].shape[1] > 1:
            axes[1].plot(time_seconds, result["x"][:, 1], label=r"$V_{C1}$")
        axes[1].legend(loc="lower left")
        axes[1].grid()

        axes[2].set_title("Low-Pass Filtered Innovation")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Error [V]")
        axes[2].plot(time_seconds, innovation_filtered, label="Low-pass Filtered Innovation")
        axes[2].legend(loc="lower left")
        axes[2].grid()

        pyplot.show()


if __name__ == "__main__":
    main()