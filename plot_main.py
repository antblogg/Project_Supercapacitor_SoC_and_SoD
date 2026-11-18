
import time

import numpy as np
from parameters import Parameters
import csv

from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

from csvInterface import getPowerProfile
from csvInterface import getVoltageProfile
from csvInterface import getCurrentProfile
from csvInterface import getDataColumn
from circuit.Circuit import Circuit


class Application:

    def __init__(self, plot = True):

        self.presetPlotLimits = False
        self.instantPlot = False

        # plot setup
        self.fig, self.ax = pyplot.subplots()
        self.ax.grid()
        self.lines = []
        for _ in range(10):
            self.lines.append(self.ax.plot([0])[0]) # plot nothing to create a plot
        if plot:
            self.updatePlot()
        


    def updatePlot(self) -> None:
        P = getPowerProfile()
        I = getCurrentProfile()
        v_ref = getVoltageProfile()
        clamp  = getDataColumn("clamp_active")

        circuit = Circuit(Parameters.RC_ORDER, v_ref[0])

        V = np.zeros(len(v_ref))
        SOC = np.zeros(len(v_ref))
        SOD = np.zeros(len(v_ref))
        V = np.zeros(len(v_ref))
        SOD_cutoff_line = np.full(len(v_ref),190)
        P = getPowerProfile()
        for i in range(len(P)):
            V[i] = circuit.computeNextState(P[i]) 
            SOC[i] = circuit.calculateSocPercent()
            SOD[i] = circuit.calculateSod() if i % 20 ==1 else SOD[i-1] # skip some timesteps to save compute time
        self.ax.legend(["Simulated voltage", "Measured voltage"])
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Terminal voltage [V]")
        
        self.drawPlot([V, v_ref], timePeriod=7)

    def renderFrame(self,frame,Y, samplesPerFrame):
        currentIdx = frame*samplesPerFrame
        x = np.arange(len(Y[0]))*Parameters.DT

        for i, y in enumerate(Y):
            self.lines[i].set_data(x[:currentIdx], y[:currentIdx])
        if not self.presetPlotLimits:
            self.ax.relim()
            self.ax.autoscale()

        return self.lines


    def drawPlot(self, Y: list[np.array], timePeriod: float = 5, keepOpen = True):
        """
        Draw the plot canvas over a given time period [s].

        :param Y: LIST of all lines to plot. For a single line y, use [y] as the argument.
        :param timePeriod: How many seconds it should take to plot the figure.
        """
        if self.presetPlotLimits:
            x = np.arange(len(Y[0]))*Parameters.DT
            for i in range(len(Y)):
                self.lines[i].set_data(x, Y[i])
            self.ax.relim()
            self.ax.autoscale()
            self.ax.grid()
        if self.instantPlot:
            timePeriod = 0.1

        self.ax.get_legend().set_loc("lower left")
        dt_ms = 1000*Parameters.DT

        samplesPerFrame = int(len(Y[0])  *dt_ms / timePeriod / 1000)

        ani = FuncAnimation(
            self.fig,
            self.renderFrame,
            frames=int(len(Y[0])/samplesPerFrame),
            interval=dt_ms,
            fargs=(Y, samplesPerFrame),
            repeat=False
        )
        pyplot.show(block=keepOpen)


def main():

    app = Application()


# Import guard:
# Run main only if this is the file being executed
if __name__ == "__main__":
    main()