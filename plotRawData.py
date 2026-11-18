import numpy as np
import csv
from parameters import Parameters
from typing import Optional
from pathlib import Path

DATASETS = (
    ("assets/30.10.2026/fullChargeDischarge_processed_full.csv", "F"),
    ("assets/02.10.2026/usn-supercap_green-profile_no-recharge_partial_2026-09-25_processed_full.csv", "D"),
    ("assets/09.10.2026/2026-10-09_binabdu_run_1_processed_full.csv", "C"),
    ("assets/30.10.2026/power_profile_20kw_processed_full.csv", "E"),
    ("assets/09.10.2026/2026-10-09_binabdu_v2_processed_full.csv", "A"),
    ("assets/09.10.2026/2026-10-09_fredrik_2_processed_full.csv", "B"),
)

PLOT_MODES = (
    ("minimal", False, False, False),
    ("full", True, True, True),
    ("power_only", False, False, True),
)

display_plots = False



def getDataColumn(columnName: str, filepath: Optional[str] = None)->list[float | str]:
    """
    Returns the data column given by columnName.
    Raises IndexError if the column is not found in the file.

    :param columnName: The name of the column to be fetched
    :param filepath: The filepath to fetch sample from. defaults to Parameters.DATA_FILEPATH.

    :returns column: list of all elements in the column. Float preferred, falls back to string.

    :raises IndexError: Column not in file
    """
    currentFile = Path(__file__)
    rootDirectory = currentFile.parents[0] # .parents[N] if file not in root directory
    if filepath is None:
        filepath = Parameters.DATA_FILEPATH
    filepath = rootDirectory / filepath
    with open(filepath,"r") as file:
        reader = csv.DictReader(file)
        if columnName not in reader.fieldnames:
            raise IndexError(f"\"{columnName}\" not in file {filepath}") 
        try:
            allRows = [float(row[columnName]) for row in reader]
        except ValueError:
            allRows = [str(row[columnName]) for row in reader]
        return allRows
    
    
def getTimeSteps(filepath: Optional[str] = None) -> np.array:
    try:
        return np.array(getDataColumn("time", filepath)) 
    except IndexError:
        return np.array(getDataColumn("ts", filepath)) # alternate name convention

def getPowerProfile(filepath: Optional[str] = None) -> np.array:
    try:
        return np.array(getDataColumn("grid_w", filepath))
    except IndexError:
        return np.array(getDataColumn("P_grid_W", filepath)) # old name convention - undocumented 

def getVoltageProfile(filepath: Optional[str] = None) -> np.array:
    try: return np.array(getDataColumn("sc_v", filepath))
    except IndexError: pass
    try: return np.array(getDataColumn("V_sc_V", filepath)) # old name convention - undocumented 
    except IndexError: pass
    print("GetVoltageProfile> Waring! Supercap voltage not found. Falling back to grid voltage")
    try: return np.array(getDataColumn("grid_v", filepath))
    except IndexError: 
        return np.array(getDataColumn("V_grid_V", filepath)) # old name convention - undocumented 

def getCurrentProfile(filepath: Optional[str] = None) -> np.array:
    try: return np.array(getDataColumn("sc_a_signed", filepath))
    except IndexError: pass
    print("GetCurrentProfile> Waring! Signed current not found. Falling back to unsigned current")
    try: return np.array(getDataColumn("sc_a", filepath))
    except IndexError: pass
    try: return np.array(getDataColumn("I_sc_A", filepath)) # old name convention - undocumented 
    except IndexError: pass
    print("GetCurrentProfile> Waring! Supercap current not found. Falling back to grid current")
    try: return np.array(getDataColumn("mean_a", filepath)) 
    except IndexError: 
        return np.array(getDataColumn("I_grid_A", filepath)) # old name convention - undocumented 





def normalizeArray(A: list | np.ndarray) -> np.ndarray:
    """
    Normalize an array by subtracting the mean
    and dividing by the standard deviation.
    Useful for plotting values with different units
    
    :param A: Array to be normalized

    :returns Normalized: normalized array
    """
    A = np.array(A)
    m = np.mean(A)
    std = np.std(A)
    A = (A - m) / std
    return A


from circuit.Circuit import Circuit
from matplotlib import pyplot
import matplotlib
def main():
    output_dir = Path(__file__).parent / "output" / "plotRawData"
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode_name, plot_simulated_voltage, plot_simulated_current, plot_ordered_power in PLOT_MODES:
        mode_dir = output_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)

        for filepath, dataset_id in DATASETS:
            Parameters.DATA_FILEPATH = filepath

            P = getPowerProfile()
            P_ordered = np.array(getDataColumn("delta_power")) * 22
            V = getVoltageProfile()
            I = getCurrentProfile()

            N = len(P)
            V_sim = np.zeros(N)
            I_sim = np.zeros(N)

            circuit = Circuit(2, V[0])
            for i in range(N):
                V_sim[i] = circuit.computeNextState(P[i])
                I_sim[i] = P[i] / V_sim[i]

            time = getTimeSteps()
            P_kw = P / 1000

            fig, (ax0, ax1, ax2) = pyplot.subplots(
                3, 1, figsize=(13, 9), dpi=100, sharex=True
            )

            fig.suptitle(f"Dataset {dataset_id}")

            ax0.set_title("Voltage")
            ax0.plot(time, V, label="Measured Voltage")
            if plot_simulated_voltage:
                ax0.plot(time, V_sim, label="Simulated Voltage")
            ax0.set_xlabel("Time (s)")
            ax0.set_ylabel("Voltage (V)")
            ax0.grid(True)
            ax0.legend()

            ax1.set_title("Current")
            ax1.plot(time, I, label="Measured Current")
            if plot_simulated_current:
                ax1.plot(time, I_sim, label="Simulated Current")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Current (A)")
            ax1.grid(True)
            ax1.legend()

            ax2.set_title("Grid Power")
            ax2.plot(time, P_kw, label="Measured Power")
            if plot_ordered_power:
                ax2.plot(time, P_ordered, label="Requested Power")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Power (kW)")
            ax2.grid(True)
            ax2.legend()

            fig.tight_layout()
            fig.savefig(mode_dir / f"{dataset_id}.png", dpi=300)
            pyplot.close(fig)

    if display_plots:
        pyplot.show()
if __name__ == "__main__":
    main()