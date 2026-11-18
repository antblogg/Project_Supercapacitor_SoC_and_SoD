from parameters import Parameters

class CircuitBranch:
    def __init__(self, R: float, C: float, V0: float)-> None:
        """
        Initialize a circuit branch, as per docs/CircuitBranch.png

        :param: R: Value of branch resistor
        :param: C: Value of branch capacitor
        :param: V0: Initial voltage of capacitor
        
        """
        self.R = R
        self.C = C
        self.V = V0

    def computeCapVoltage(self, Vt: float) -> float:
        """
        Compute the voltage of the capacitor
        given the current terminal voltage.

        :param Vt: Terminal voltage to be applied.

        :returns Vn: computed voltage on this capacitor
        """
        # using Ohms law. voltage over resistor = difference
        # between terminal voltage and cap voltage
        i = (Vt-self.V)/self.R
        dVdt = i/self.C
        V = self.V + dVdt * Parameters.DT
        return V