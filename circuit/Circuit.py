from circuit.CircuitBranch import CircuitBranch
from parameters import Parameters 
from typing import Optional
import numpy as np
from warnings import warn

class Circuit:
    def __init__(self, SystemRcOrder: Optional[int] = None, InititalVoltage: Optional[float] = None) -> None:
        """
        Initalize a N-order RC parallel circuit.

        :param SystemRcOrder: Order of the system. 0 = RC-0, 1=RC-1 and so on
        :param InitialVoltage: The voltage to initialize all capacitors to.
        """
        # initialize variables
        if SystemRcOrder is None: SystemRcOrder = Parameters.RC_ORDER
        if InititalVoltage is None: InititalVoltage = Parameters.InitialTerminalVoltage
        self.systemOrder = SystemRcOrder 
        self.circuitBranches: list[CircuitBranch] = []
        self.V0 = InititalVoltage
        self.V = self.V0
        self.R_L = Parameters.R_L
        
        # add branches according to the system order.
        for i in range(SystemRcOrder+1):
            R, C, V0_val = self.getCircuitParametersByNumber(i)
            #  branch starts exactly at the system InitialTerminalVoltage
            self.saveBranch(CircuitBranch(R, C, self.V0))

        self.maxSoc = sum(
            C*V
            for C, V in zip(
                [b.C for b in self.circuitBranches],
                [400]*len(self.circuitBranches)
                ))
        

    def setLeakageResistance(self, R_L):
        """
        Set the leakage resistance. (This function mainly exists to guide llms tbh)
        """
        self.R_L = R_L

    def _calculateSoc(self) -> float:
        soc = 0
        for branch in self.circuitBranches:
            soc +=branch.V*branch.C
        return soc

    def calculateSocPercent(self) -> float:
        """
        Returns SOC in percent: [0,1]
        """
        currentSoc = self._calculateSoc()
        maxSoc = self.maxSoc
        return currentSoc/maxSoc

    def calculateSod(self, P_requested = 22_000):
        # record State
        oldVoltage = self.V
        oldBranchVoltages = [b.V for b in self.circuitBranches]

        # Simulate until done.
        t = 0
        while self.V > 0 and self.V < Parameters.V_MAX and abs(P_requested/self.V) < Parameters.I_MAX:
            self.computeNextState(P_requested)
            t+=Parameters.DT
        # Reset state
        self.V = oldVoltage
        for branch, old_V in zip(self.circuitBranches,oldBranchVoltages):
            branch.V = old_V
        return t

    def computeNextState(self, P_target: float):
        """
        Update the internal state of the RC-model smoothly with safety voltage limits.

        :param P_target: The desired Power P for the current timestep.
        :returns V: voltage of the system at the current time.
        """
        # 1. Compute the terminal current needed based on the previous step's voltage
        currentSupply = np.divide(np.float64(P_target), np.float64(self.V)) if self.V != 0 else 0.0
        #currentSupply = P_target
        
        # 2. Compute the true physical instantaneous terminal voltage 
        V = self.computeTerminalVoltage(currentSupply)
        
        # 3. SAFETY ENFORCEMENT: Ensure the voltage between V_MIN and V_MAX limits
        if V < Parameters.V_MIN or V > Parameters.V_MAX:
            warn("Voltage out of bounds")
        else:
            self.V = V
        
        # 4. Update the internal charge state for the NEXT step
        for branch in self.circuitBranches:
            branch.V = branch.computeCapVoltage(self.V)
    
        return self.V 

    def computeTerminalVoltage(self, It) -> float:
        """
        Compute the terminal voltage given
        the terminal current. See the readme doc
        for how these equations were derived.

        :param It: The current requested for this timestep.
        :returns Vt: Terminal voltage given the requested current.
        """
        sum_one_divBy_R = 0
        sum_V_divBy_R = 0
        
        for branch in self.circuitBranches:
            sum_one_divBy_R += 1 / branch.R
            sum_V_divBy_R += branch.V / branch.R
            
        numerator = sum_V_divBy_R - It 
        denominator = sum_one_divBy_R + 1/self.R_L
        
        V_terminal = numerator / denominator
        return V_terminal
    
    def reset(self):
        self.V = self.V0
        for i in range(len(self.circuitBranches)):
            branch = self.circuitBranches[i]
            branch.V = self.V0

    def saveBranch(self, component: CircuitBranch) -> None:
        self.circuitBranches.append(component)

    @staticmethod # <-- means "this function doesn't need self as an input"
    def getCircuitParametersByNumber(i):
        R = Parameters.R[i]
        C = Parameters.C[i]
        V0_val = Parameters.InitialTerminalVoltage
        return R, C, V0_val
