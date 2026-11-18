
"""
Script to bundle raw data where each zeno tag is published seperately
into a single csv file with columns for each tag and a time column.

Input the filename of the raw csv file in the varibles below.

Outputs two files in the same folder as the input file:
1. <filename>_processed_full.csv - All the data collected
2. <filename>_processed_short.csv - A subset of the most relevant columns
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
# dont touch this - path starts from the location of this script.

# add the path of the folder where the csv file is" is.
# use ".parent" to go back one level in the directory structure.
CSV_FOLDER = Path(__file__).parent 

    # Sampling time in seconds
    # should be large enough to make sure that every column is loaded at least once per DT second.
DT = 0.05  

# optionally delete columns that are not needed in the final output.
DROP_COLUMNS = [
        #"time",
        #"grid_var",
        #"grid_w",
        #"sc_v",
        #"play",
        #"ctrl_applied",
        #"ctrl",
        #"sc_a",
        #"grid_v",
        #"mean_a",
        #"clamp_active",
        #"dc_v",
        #"i_dir",
        #"id",
        #"ref_pu",
    ]


def calculateSign(P_grid):
    threshold = 130
    sign = np.zeros(len(P_grid))
    for i in range(len(sign)):
        if P_grid[i] > 0: sign[i] = 1
        if P_grid[i] < 0: sign[i] = -1
        if abs(P_grid[i]) < threshold: sign[i] = 1
    return sign




def read_csv_file(file_path: str | Path) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)


def main() -> None:
    for file in os.listdir(CSV_FOLDER):
        if not file.endswith(".csv"): continue
        FILENAME = file # writing ".csv" is optional
        FILENAME = FILENAME.split(".")[0]
        CSV_PATH = CSV_FOLDER / (FILENAME+".csv")
        topics = {
            "phase":"telemark_hybrid/usn_supercap/agent/status/play",
            "grid_var": "telemark_hybrid/usn_supercap/meter/out/Q/grid_var",
            "grid_w": "telemark_hybrid/usn_supercap/meter/out/P/grid_w",
            "sc_v": "telemark_hybrid/usn_supercap/plc/out/V/sc_v",
            "play": "telemark_hybrid/usn_supercap/agent/status/play",
            "ctrl_applied": "telemark_hybrid/usn_supercap/chess/out/y/ctrl_applied",
            "ctrl": "telemark_hybrid/usn_supercap/power_ctrl/out/y/ctrl",
            "sc_a": "telemark_hybrid/usn_supercap/plc/out/I/sc_a",
            "grid_v": "telemark_hybrid/usn_supercap/meter/out/V/grid_v",
            "mean_a": "telemark_hybrid/usn_supercap/meter/out/I/mean_a",
            "clamp_active": "telemark_hybrid/usn_supercap/chess/state/clamp_active",
            "dc_v": "telemark_hybrid/usn_supercap/plc/out/V/dc_v",
            "i_dir": "telemark_hybrid/usn_supercap/chess/out/y/i_dir",
            "id": "telemark_hybrid/usn_supercap/chess/state/id",
            "ref_pu": "telemark_hybrid/usn_supercap/chess/out/P/ref_pu",
            "delta_power": "telemark_hybrid/usn_supercap/chess/cmd/delta_power",
            "soe": "telemark_hybrid/usn_supercap/chess/status/soe",
        }


        data: pd.DataFrame = read_csv_file(CSV_PATH)

        df = data.copy()

        df["ts"] = pd.to_datetime(df["ts"], format="mixed", utc=True)
        # Convert anything non-numeric to NaN
        agent_df = df[df["topic"] == topics["phase"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # Remove rows where conversion failed
        df = df.dropna(subset=["value"])
        agent_pivot = agent_df.pivot_table(
        index="ts",
        columns="topic",
        values="value",
        aggfunc="last"
        )
        agent_resample = (
            agent_pivot
            .resample(f"{int(DT * 1000)}ms")
            .last()
            .ffill()
        )
        pivot = df.pivot_table(
        index="ts",
        columns="topic",
        values="value",
        aggfunc="last"
        )

        resampled = (
            pivot
            .resample(f"{int(DT * 1000)}ms")
            .last()
            .ffill()
        )
        resampled = resampled.iloc[100:].reset_index(drop=True)
        agent_resample = agent_resample.iloc[100:].reset_index(drop=True)
        agent_resample.rename(inplace=True, columns={"phase": topics["phase"]})

        if agent_resample.values.size > 0:
            resampled = resampled.join(agent_resample)
        
        resampled.rename(inplace=True, columns={v:k for k,v in topics.items()})
        resampled.insert(0, "time", np.arange(len(resampled)) * DT)
        sign = calculateSign(resampled["grid_w"])
        signedCurrent = resampled["sc_a"] * sign
        resampled["sc_a_sign"] = sign
        resampled["sc_a_signed"] = signedCurrent
        for col in DROP_COLUMNS:
            try:
                resampled.drop(
                columns=[col],
                inplace=True,
                )
            except KeyError as e:
                print(f"Column not found: {e}")
        
        short_data = resampled[
            [
                "time",
                "grid_w",
                "sc_v",
                "sc_a",
                "sc_a_signed",
                "ctrl",
                "delta_power",
            ]
        ].copy()
        resampled.to_csv(CSV_PATH.parent / (FILENAME + "_processed_full.csv"), index=False)
        short_data.to_csv(CSV_PATH.parent / (FILENAME+"_processed_short.csv"), index=False)
    print("Processed CSV files saved successfully to:")
    print(CSV_PATH.parent)
if __name__ == "__main__":
    main()
