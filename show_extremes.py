import numpy as np
from matplotlib import pyplot

from csvInterface import getCurrentProfile, getVoltageProfile, getDataColumn
from parameters import Parameters


FILEPATH = None  # None: use parameters.py
FILEPATH = "assets/30.10.2026/fullChargeDischarge_processed_full.csv"

# Adjust these limits to change the zoomed views.
VOLTAGE_ZOOM_XLIM = (70, 110)
VOLTAGE_ZOOM_YLIM = (340, 420)
CURRENT_ZOOM_XLIM = (145, 165)
CURRENT_ZOOM_YLIM = (140, 210)
SHOW_CLAMP_ACTIVE = False

if not hasattr(Parameters, "I_MAX"):
    Parameters.I_MAX = 190
if not hasattr(Parameters, "V_MIN"):
    Parameters.V_MIN = 1e-5


def _parse_clamp_value(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true"):
            return 1
        if normalized in ("0", "false"):
            return 0
        raise ValueError(f"Unrecognized clamp_active value: {value!r}")
    return int(bool(value))


def main():
    if FILEPATH is not None:
        Parameters.DATA_FILEPATH = FILEPATH

    voltage_reference = getVoltageProfile()
    current_reference = getCurrentProfile()
    voltage_time = np.arange(len(voltage_reference)) * Parameters.DT
    current_time = np.arange(len(current_reference)) * Parameters.DT

    fig, axes = pyplot.subplots(
        2,
        2,
        figsize=(10, 6),
        constrained_layout=True,
    )
    voltage_ax, current_ax = axes[0]
    voltage_zoom_ax, current_zoom_ax = axes[1]

    voltage_line = f"Maximum voltage ({Parameters.V_MAX:g} V)"
    current_line = f"Current limit ({Parameters.I_MAX:g} A)"
    for ax in (voltage_ax, voltage_zoom_ax):
        ax.plot(voltage_time, voltage_reference, color="tab:blue", label="Reference")
        ax.axhline(
            Parameters.V_MAX,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=voltage_line,
        )
        ax.set_ylabel("Voltage [V]")
        ax.grid(True, alpha=0.25)

    for ax in (current_ax, current_zoom_ax):
        ax.plot(current_time, current_reference, color="tab:green", label="Reference")
        ax.axhline(
            Parameters.I_MAX,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=current_line,
        )
        ax.set_ylabel("Current [A]")
        ax.grid(True, alpha=0.25)

    if SHOW_CLAMP_ACTIVE:
        clamp_values = getDataColumn("clamp_active")
        clamp_active = np.asarray(
            [_parse_clamp_value(value) for value in clamp_values],
            dtype=int,
        )
        clamp_time = np.arange(len(clamp_active)) * Parameters.DT

        for ax in axes.flat:
            clamp_ax = ax.twinx()
            clamp_line, = clamp_ax.step(
                clamp_time,
                clamp_active,
                where="post",
                color="black",
                linestyle="-",
                linewidth=1.8,
                alpha=0.9,
                label="Clamp active",
            )
            clamp_ax.set_ylim(-0.1, 1.1)
            clamp_ax.set_yticks((0, 1), labels=("Off", "On"))
            clamp_ax.tick_params(axis="y", colors="0.4", labelsize=8)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles + [clamp_line], labels + ["Clamp active"])
    else:
        for ax in axes.flat:
            ax.legend(loc="lower right")

    voltage_ax.set_title("Voltage profile")
    current_ax.set_title("Current profile")
    voltage_zoom_ax.set_title("Voltage profile - zoomed")
    current_zoom_ax.set_title("Current profile - zoomed")

    voltage_zoom_ax.set_xlim(VOLTAGE_ZOOM_XLIM)
    if VOLTAGE_ZOOM_YLIM is not None:
        voltage_zoom_ax.set_ylim(VOLTAGE_ZOOM_YLIM)

    current_zoom_ax.set_xlim(CURRENT_ZOOM_XLIM)
    if CURRENT_ZOOM_YLIM is not None:
        current_zoom_ax.set_ylim(CURRENT_ZOOM_YLIM)

    for ax in axes[1]:
        ax.set_xlabel("Time [s]")

    fig.suptitle("Reference voltage and current profiles")
    pyplot.show()


if __name__ == "__main__":
    main()
