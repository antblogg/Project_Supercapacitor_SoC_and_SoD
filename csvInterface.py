import numpy as np
import csv
from parameters import Parameters
from typing import Optional
from pathlib import Path


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

def getPowerProfile(filepath: Optional[str] = None) -> np.array:
    try:
        return getCurrentProfile(filepath) * getVoltageProfile(filepath)
    except IndexError:
        return np.array(getDataColumn("P_grid_W", filepath)) # old name convention - undocumented
    """
    try:
        return np.array(getDataColumn("grid_w", filepath))
    except IndexError:
        return np.array(getDataColumn("P_grid_W", filepath)) # old name convention - undocumented 
    """



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
from matplotlib import pyplot as plt
import matplotlib
def main():

    V = getVoltageProfile()
    I = getCurrentProfile()
    
    circuit = Circuit(InitialVoltage = 250)
    P = getPowerProfile()
    N = len(P)
    V_sim = np.zeros(N)
    for i in range(N):
        V_sim[i] = circuit.computeNextState(P[i])
    plt.plot(V_sim)

    P_norm = normalizeArray(P)
    V_norm  = normalizeArray(V)
    I_norm  = normalizeArray(I)
    V_sim_norm  = normalizeArray(V_sim)

    fig = pyplot.figure(figsize=(13, 7), dpi=100)
    #fig.tight_layout()
    ax0 = fig.add_subplot(2,2,(3,4))
    ax1 = fig.add_subplot(2,2,(1))
    ax2 = fig.add_subplot(2,2,(2))

    ax0.set_title("Voltage - Measured VS simulated")
    ax0.set_xticks([])
    ax0.grid(True)
    ax0.plot(V)
    ax0.plot(V_sim)
    ax0.legend(["V_t", "V_sim"])
    
    ax1.set_title("Normalized Power VS voltage")
    ax1.grid(True)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.plot(P_norm)
    ax1.plot(V_sim_norm)
    ax1.legend(["P", "V_sim"])

    ax2.set_title("Normalized Power VS Current (unsigned)")
    ax2.grid(True)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.plot(P_norm)
    ax2.plot(I_norm)
    ax2.legend(["P", "I"])

    pyplot.show()

if __name__ == "__main__":
    main()