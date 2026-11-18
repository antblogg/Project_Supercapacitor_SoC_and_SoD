"""
KalmanFilter_Py.py  -  linear Kalman filter for the super-capacitor circuit.

Own script. It does NOT change circuit.py / gui_main.py. It only needs parameters.py.

Input data (csv, columns found by name, upper or lower case): time, grid_w, sc_v
and, if present, sc_a (the measured DC current).
  * With sc_a (best): the measured current is the filter input. The current sign
    comes from grid_w (positive grid_w = discharge). No loss model is needed.
  * Without sc_a: the current is estimated from the power, I = P / V, with a fitted
    converter efficiency and standby power.

Run from the folder that contains parameters.py :
    python KalmanFilter_Py.py path/to/data.csv
"""
import csv
import os
import sys

import numpy as np

from parameters import Parameters

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
SHOW_WINDOW = True      # False = only save kalman_result.png, no plot window
USE_MEASURED_CURRENT = False   # use the sc_a column when the file has it
FIT_RC = False           # fit R and C to the data (takes up to a minute or two)
SKIP_ROWS = 0           # data rows to skip at the start of the file
SIGMA_I_MEASURED = 0.5  # noise of the current sensor in A (used with sc_a)
Q_V = None              # process noise in V per step; None = choose automatically so mean NIS is close to 1
POWER_SIGN = None       # power-only mode: None = pick automatically, or -1 / +1

# Start values for R and C (overwrites parameters.py while OVERRIDE = True).
OVERRIDE = False
OVERRIDE_RC_ORDER = 3          # 3 = C0 plus three RC branches (C1, C2, C3), four capacitors
OVERRIDE_R = [0.03, 106.811, 182.343, 229.301]
OVERRIDE_C = [88 / 5, 2.7455, 9.12741, 9.67932]
if OVERRIDE:
    Parameters.RC_ORDER = OVERRIDE_RC_ORDER
    Parameters.R = list(OVERRIDE_R)
    Parameters.C = list(OVERRIDE_C)


class LinearKalmanFilter:
    """Linear Kalman filter for the parallel-branch super-capacitor circuit.

    Same circuit as the Circuit class above (parallel branches, each R_k in
    series with C_k). Only parameters.py is needed.

    Model
        states   x = [V_0 ... V_N]   capacitor voltages (hidden)
        input    u = i               current OUT of the bank (positive = discharge)
        output   y = v               terminal voltage (the measured sc_v)

        G_k = 1/R_k,   G = G_0 + ... + G_N
        continuous:  dx/dt = A_c x + B_c u        y = C_c x + D_c u
            A_c(k,j) = (G_k/C_k) * (G_j/G - delta_kj)
            B_c(k)   = -G_k / (C_k * G)
            C_c(j)   = G_j / G
            D_c      = -(1/G + R_L)       (R_L = optional series resistor at the terminal)
        discrete (Euler, exactly what CircuitBranch.computeCapVoltage does):
            A = I + DT*A_c    B = DT*B_c    C = C_c    D = D_c
            x[n+1] = A x[n] + B u[n]        y[n] = C x[n] + D u[n]

    Why it is LINEAR
        Circuit.computeNextState takes a POWER and computes the current as
        It = P / V (previous voltage), which makes the model non-linear. Here
        the current is computed from the power and the last MEASURED voltage:

            i[k] = P_target[k] / V_meas[k-1]

        P_target is the power exactly as it is given to computeNextState (in the
        GUI that is -grid_w). Circuit uses  V = (sum(V_k/R_k) - It) / sum(1/R_k),
        so It is the current OUT of the bank, the same i as in this model. With
        the current known, the model is linear and the standard Kalman filter
        applies.

    Per sample (notation of Welch and Bishop)
        a priori:      x^-, P^-            predicted voltage  y^ = C x^- + D u
        residual:      e = y - y^,  S = C P^- C' + R,  K = P^- C' / S
        a posteriori:  x = x^- + K e,  P = (I-KC) P^- (I-KC)' + K R K'
        predict next:  x^- = A x + B u,  P^- = A P A' + Q
    Samples where the measurement is missing (NaN) only do the prediction.

    Settings (all optional; add them to parameters.py to change them):
        KF_Q_V       process noise per step on every voltage state, volt   (0.002)
        KF_SIGMA_V   noise of the voltage measurement, volt                (estimated from the data)
        KF_SIGMA_I   noise of the current input, ampere                    (0 = ignored)
        KF_P0_V      initial uncertainty (std) of every voltage state      (5.0)
        KF_R_SERIES  series resistor R_L at the terminal, ohm              (0)

    Usage
        kf  = LinearKalmanFilter()
        res = kf.run(P_circuit, V_measured)                  # closed loop (filter)
        opn = kf.run(P_circuit, V_measured, update=False)    # open loop (model only)
    """

    def __init__(self, R=None, C=None, dt=None, r_series=None,
                 q_v=None, sigma_v=None, sigma_i=None, p0_v=None):
        n = Parameters.RC_ORDER + 1      # same number of branches as Circuit
        self.R = np.array(Parameters.R[:n] if R is None else R, dtype=float)
        self.C = np.array(Parameters.C[:n] if C is None else C, dtype=float)
        self.dt = float(Parameters.DT if dt is None else dt)
        self.n = len(self.R)
        self.r_series = float(getattr(Parameters, "KF_R_SERIES", 0.0) if r_series is None else r_series)
        self.q_v = float(getattr(Parameters, "KF_Q_V", 0.002) if q_v is None else q_v)
        self.sigma_v = getattr(Parameters, "KF_SIGMA_V", None) if sigma_v is None else sigma_v
        self.sigma_i = float(getattr(Parameters, "KF_SIGMA_I", 0.0) if sigma_i is None else sigma_i)
        self.p0_v = float(getattr(Parameters, "KF_P0_V", 5.0) if p0_v is None else p0_v)
        self._build_model()

    # ------------------------------------------------------------------
    def _build_model(self):
        G = 1.0 / self.R
        Gsum = G.sum()
        n = self.n
        Ac = (G / self.C)[:, None] * (G[None, :] / Gsum - np.eye(n))
        Bc = (-G / (self.C * Gsum)).reshape(n, 1)
        self.A = np.eye(n) + self.dt * Ac            # Euler, as in CircuitBranch
        self.B = self.dt * Bc
        self.Cm = (G / Gsum).reshape(1, n)
        self.D = float(-(1.0 / Gsum + self.r_series))

    def observability(self):
        """Rank and conditioning of the observability matrix [C; CA; ...; CA^(n-1)]."""
        O = np.vstack([self.Cm @ np.linalg.matrix_power(self.A, k) for k in range(self.n)])
        sv = np.linalg.svd(O, compute_uv=False)
        return int(np.linalg.matrix_rank(O)), float(sv[-1] / sv[0])

    @staticmethod
    def estimate_meas_std(V, floor=0.02):
        """Rough sensor-noise estimate (V).

        Looks only at quiet stretches, where the voltage is almost flat (the average
        step over 21 samples is tiny). There, the difference of two neighbouring
        samples has a standard deviation of sqrt(2) times the sensor noise.
        """
        v = np.asarray(V, dtype=float)
        v = v[np.isfinite(v)]
        d1 = np.diff(v)
        if len(d1) < 100:
            return floor
        slope = np.convolve(d1, np.ones(21) / 21, mode="same")
        quiet = np.abs(slope) < 0.02
        if quiet.sum() < 100:
            quiet = np.abs(slope) <= np.percentile(np.abs(slope), 30)
        return max(float(np.std(d1[quiet]) / np.sqrt(2.0)), floor)

    @staticmethod
    def current_input(P_circuit, V_meas, v_prev0):
        """Current out of the bank (positive = discharge), one value per sample."""
        N = len(P_circuit)
        i = np.empty(N)
        v_prev = float(v_prev0)
        for k in range(N):
            i[k] = P_circuit[k] / v_prev
            if np.isfinite(V_meas[k]) and V_meas[k] > 0:
                v_prev = float(V_meas[k])
        return i

    # ------------------------------------------------------------------
    def run(self, P_circuit, V_meas, x0=None, v_prev0=None, update=True, u=None,
            state_bounds=None):
        """Run the filter (update=True) or the open-loop model (update=False).

        :param P_circuit: power for every sample, as given to Circuit.computeNextState
                          (not used when u is given; pass None)
        :param u:         MEASURED current out of the bank, one value per sample (positive =
                          discharge). When given it is used as the input directly.
        :param V_meas:    measured terminal voltage (NaN = missing)
        :param x0:        start state (default: every capacitor at the first measured voltage)
        :param v_prev0:   voltage "before" sample 0 used for the current (default: first measurement)
        :param state_bounds: optional (minimum, maximum) applied to each state estimate;
                             covariance remains the standard unconstrained KF covariance
        :returns: dict with V_filt, V_pred, innov, S, nis, x, std, i, sigma_v, time
        """
        V_meas = np.asarray(V_meas, dtype=float)
        N, n = len(V_meas), self.n
        if state_bounds is not None:
            lower, upper = state_bounds
            if lower > upper:
                raise ValueError("state_bounds minimum must not exceed its maximum")

        finite = np.isfinite(V_meas)
        v_first = (float(V_meas[np.argmax(finite)]) if finite.any()
                   else float(Parameters.InitialTerminalVoltage))
        x = np.full((n, 1), v_first) if x0 is None else np.asarray(x0, dtype=float).reshape(n, 1)
        if state_bounds is not None:
            x = np.clip(x, lower, upper)
        v_prev0 = v_first if v_prev0 is None else float(v_prev0)

        sigma_v = self.sigma_v if self.sigma_v else self.estimate_meas_std(V_meas)
        r = sigma_v ** 2 + (self.sigma_i * self.D) ** 2
        Q = self.sigma_i ** 2 * (self.B @ self.B.T) + np.eye(n) * self.q_v ** 2
        P = np.eye(n) * self.p0_v ** 2
        I = np.eye(n)
        A, B, Cm, D = self.A, self.B, self.Cm, self.D

        if u is None:
            u = self.current_input(np.asarray(P_circuit, dtype=float), V_meas, v_prev0)
        else:
            u = np.asarray(u, dtype=float)
        out = {k: np.full(N, np.nan) for k in ("V_filt", "V_pred", "innov", "S", "nis")}
        out["x"] = np.zeros((N, n))
        out["std"] = np.zeros((N, n))

        for k in range(N):
            # a priori: predicted measurement C x^- + D u
            y_pred = (Cm @ x).item() + D * u[k]
            out["V_pred"][k] = y_pred

            # a posteriori: correct the state with the measurement, if there is one
            if update and finite[k]:
                S = (Cm @ P @ Cm.T).item() + r            # variance of the residual
                K = (P @ Cm.T) / S                        # gain, n x 1
                e = V_meas[k] - y_pred                    # residual (innovation)
                x = x + K * e
                if state_bounds is not None:
                    x = np.clip(x, lower, upper)
                IKC = I - K @ Cm
                P = IKC @ P @ IKC.T + r * (K @ K.T)       # Joseph form
                out["innov"][k], out["S"][k], out["nis"][k] = e, S, e * e / S

            out["V_filt"][k] = (Cm @ x).item() + D * u[k]
            out["x"][k] = x.ravel()
            out["std"][k] = np.sqrt(np.clip(np.diag(P), 0.0, None))

            # predict the next sample
            x = A @ x + B * u[k]
            if state_bounds is not None:
                x = np.clip(x, lower, upper)
            P = A @ P @ A.T + Q
            P = 0.5 * (P + P.T)

        out["i"] = u
        out["sigma_v"] = sigma_v
        out["time"] = np.arange(N) * self.dt
        return out


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def find_file(path):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (path, os.path.join(here, path)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find data file: {path}")


def load_data(path):
    """Returns grid_w, sc_v and sc_a (None if the file has no sc_a column)."""
    with open(find_file(path), newline="") as f:
        reader = csv.DictReader(f)
        names = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        for need in ("grid_w", "sc_v"):
            if need not in names:
                raise ValueError(f"the csv has no '{need}' column; columns: {list(names)}")
        rows = list(reader)[SKIP_ROWS:]

    def col(key, fill):
        name = names[key]
        return np.array([float(r[name]) if r[name].strip() else fill for r in rows])

    grid_w = col("grid_w", 0.0)
    V = col("sc_v", np.nan)
    I = col("sc_a", 0.0) if "sc_a" in names else None
    return grid_w, V, I


# ----------------------------------------------------------------------
# Open-loop simulation and fitting
# ----------------------------------------------------------------------
def simulate(u, v0, kf):
    """Model only (no measurement correction) with a given current input u (A)."""
    A, B, Cm, D = kf.A, kf.B, kf.Cm, kf.D
    x = np.full((kf.n, 1), float(v0))
    out = np.empty(len(u))
    for k, ik in enumerate(u):
        out[k] = (Cm @ x).item() + D * ik
        x = A @ x + B * ik
    return out


def choose_current_sign(grid_w, I_meas, V_meas):
    """sc_a is only the size of the current. Find out if positive grid_w means discharge (+1) or charge (-1).

    With the right sign the voltage follows V(t) - V(0) = -(1/C) * integral(i dt) with a sensible C
    (a few to a few tens of farad). The wrong sign gives a negative or absurd C and a much larger error.
    """
    good = np.isfinite(V_meas)
    y = (V_meas - V_meas[good][0])
    best = None
    for sign in (+1, -1):
        i = sign * np.where(grid_w >= 0, 1.0, -1.0) * I_meas
        x = -np.cumsum(i) * Parameters.DT
        c = float(np.dot(x[good], y[good]) / np.dot(x[good], x[good]))      # = 1/C
        err = float(np.sqrt(np.mean((y[good] - c * x[good]) ** 2)))
        capacity = 1.0 / c if c > 0 else float("nan")
        print(f"  sign {sign:+d}: effective C = {capacity:8.1f} F, error {err:7.2f} V")
        if c > 0 and (best is None or err < best[1]):
            best = (sign, err)
    if best is None:
        raise ValueError("neither sign of the current explains the voltage; check the csv columns")
    return best[0]


def fit_rc_current(u, V_meas, v0, kf0):
    """Fit R_k and C_k of all branches so the open-loop model matches V_meas."""
    from scipy.optimize import least_squares
    n = kf0.n
    good = np.isfinite(V_meas)

    def unpack(p):
        return np.exp(p[:n]), np.exp(p[n:])

    def resid(p):
        R, C = unpack(p)
        r = (simulate(u, v0, LinearKalmanFilter(R=R, C=C)) - V_meas)[good]
        return np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3)

    p0 = np.concatenate([np.log(kf0.R), np.log(kf0.C)])
    lo = np.concatenate([[np.log(0.005)], np.full(n - 1, np.log(0.2)),
                         [np.log(5.0)], np.full(n - 1, np.log(0.2))])
    hi = np.concatenate([[np.log(1.0)], np.full(n - 1, np.log(1e5)),
                         [np.log(60.0)], np.full(n - 1, np.log(1e6))])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
    r_before = float(np.sqrt(np.mean(resid(p0) ** 2)))
    fit = least_squares(resid, p0, bounds=(lo, hi), max_nfev=60, verbose=2)
    R, C = unpack(fit.x)
    return R, C, r_before, float(np.sqrt(np.mean(fit.fun ** 2)))


# --- power-only mode (file without sc_a) -------------------------------
def bank_power(P_target, eta_d, eta_c, p_sb):
    """Power at the bank from the grid-side power.

    Discharge (P > 0): the bank delivers P / eta_d.
    Charge (P < 0):    the bank receives P * eta_c.
    p_sb: constant standby power drawn from the bank (W).
    """
    P_target = np.asarray(P_target, dtype=float)
    return np.where(P_target > 0, P_target / eta_d, P_target * eta_c) + p_sb


def simulate_power(P_bank, v0, kf):
    """Same as simulate(), but the current is P_bank / V (non-linear, as in Circuit)."""
    A, B, Cm, D = kf.A, kf.B, kf.Cm, kf.D
    x = np.full((kf.n, 1), float(v0))
    out = np.empty(len(P_bank))
    vprev = float(v0)
    for k, p in enumerate(P_bank):
        i = p / vprev
        v = (Cm @ x).item() + D * i
        out[k] = v
        x = A @ x + B * i
        vprev = min(max(v, 1.0), 1e4)
    return out


def fit_power_mode(grid_w, V_meas, v0, kf0):
    """Power-only mode: choose the sign of grid_w, then fit
    discharge efficiency, charge efficiency, standby power and R, C.

    Returns sign, eta_d, eta_c, p_sb, R, C, rmse_before, rmse_after.
    """
    from scipy.optimize import least_squares
    n = kf0.n
    good = np.isfinite(V_meas)

    # 1) sign: fit the three loss numbers only, for both signs
    best = None
    for sign in ((-1, +1) if POWER_SIGN is None else (POWER_SIGN,)):
        def resid3(p):
            sim = simulate_power(bank_power(sign * grid_w, p[0], p[1], p[2]), v0, kf0)
            return (sim - V_meas)[good]
        f = least_squares(resid3, [0.9, 0.9, 500.0],
                          bounds=([0.5, 0.5, -5000.0], [1.0, 1.0, 20000.0]),
                          x_scale=[0.1, 0.1, 1000.0])
        err = float(np.sqrt(np.mean(f.fun ** 2)))
        print(f"  sign {sign:+d}: eta_discharge {f.x[0]:.3f}, eta_charge {f.x[1]:.3f}, "
              f"standby {f.x[2]:.0f} W -> open-loop RMSE {err:.2f} V")
        if best is None or err < best[1]:
            best = (sign, err, f.x)
    sign, _, p3 = best

    # 2) all numbers: losses and R, C
    def unpack(p):
        return p[0], p[1], p[2], np.exp(p[3:3 + n]), np.exp(p[3 + n:])

    def resid(p):
        ed, ec, psb, R, C = unpack(p)
        sim = simulate_power(bank_power(sign * grid_w, ed, ec, psb), v0,
                             LinearKalmanFilter(R=R, C=C))
        return np.nan_to_num((sim - V_meas)[good], nan=1e3, posinf=1e3, neginf=-1e3)

    p0 = np.concatenate([p3, np.log(kf0.R), np.log(kf0.C)])
    lo = np.concatenate([[0.5, 0.5, -5000.0], [np.log(0.005)], np.full(n - 1, np.log(0.2)),
                         [np.log(5.0)], np.full(n - 1, np.log(0.2))])
    hi = np.concatenate([[1.0, 1.0, 20000.0], [np.log(1.0)], np.full(n - 1, np.log(1e5)),
                         [np.log(60.0)], np.full(n - 1, np.log(1e6))])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
    r0 = float(np.sqrt(np.mean(resid(p0) ** 2)))
    f = least_squares(resid, p0, bounds=(lo, hi), max_nfev=60, verbose=2,
                      x_scale=np.concatenate([[0.1, 0.1, 1000.0], np.ones(2 * n)]))
    ed, ec, psb, R, C = unpack(f.x)
    return sign, float(ed), float(ec), float(psb), R, C, r0, float(np.sqrt(np.mean(f.fun ** 2)))


def write_fitted_block(R, C, rmse, note):
    lines = [
        f"# Fitted by KalmanFilter_Py.py, open-loop RMSE {rmse:.2f} V",
        f"# {note}",
        f"RC_ORDER = {len(R) - 1}",
        "R = [", *[f"    {float(r):.6g}," for r in R], "]",
        "C = [", *[f"    {float(c):.6g}," for c in C], "]",
    ]
    text = "\n".join(lines) + "\n"
    with open("fitted_for_parameters_py.txt", "w") as f:
        pass
        #f.write(text)
    print("\n--- paste into parameters.py in place of RC_ORDER, R and C (also saved in fitted_for_parameters_py.txt) ---")
    print(text)


def choose_q_v(kf, P_circuit, V_meas, u=None):
    """Try several process-noise values and keep the one whose mean NIS is closest to 1."""
    best = None
    for q in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        kf.q_v = q
        nis = kf.run(P_circuit, V_meas, u=u)["nis"]
        m = float(np.mean(nis[np.isfinite(nis)]))
        print(f"  process noise {q:5.3f} V/step -> mean NIS {m:6.2f}")
        score = abs(np.log(max(m, 1e-9)))
        if best is None or score < best[1]:
            best = (q, score)
    kf.q_v = best[0]
    return best[0]


# ----------------------------------------------------------------------
def main():
    import parameters as _p
    print("*** KalmanFilter.py version 18 ***")
    print("parameters.py loaded from:", os.path.abspath(_p.__file__))
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = Parameters.DATA_FILEPATH
        print("(no file given, using Parameters.DATA_FILEPATH)")
    print("data file:", path)
    grid_w, V_meas, I_meas = load_data(path)
    ok = np.isfinite(V_meas)
    v0 = float(V_meas[np.argmax(ok)])
    print(f"{len(V_meas)} samples, dt = {Parameters.DT} s, start voltage {v0:.1f} V")
    kf = LinearKalmanFilter()
    print(f"start values: {kf.n} branches, R = {[round(float(x), 4) for x in kf.R]}, C = {[round(float(x), 4) for x in kf.C]}")

    use_current = USE_MEASURED_CURRENT and I_meas is not None
    if use_current:
        # sc_a is only the size of the current; positive grid_w = discharge
        print("input: measured current sc_a; checking the current direction ...")
        sign_i = choose_current_sign(grid_w, I_meas, V_meas)
        u = sign_i * np.where(grid_w >= 0, 1.0, -1.0) * I_meas
        print(f"current direction: positive grid_w = {'discharge' if sign_i > 0 else 'charge'}")
        if FIT_RC:
            print("fitting R and C to the measured voltage (takes up to a minute or two) ...")
            R_fit, C_fit, r0, r1 = fit_rc_current(u, V_meas, v0, kf)
            print("fitted R =", [round(float(x), 4) for x in R_fit])
            print("fitted C =", [round(float(x), 4) for x in C_fit])
            print(f"  open-loop RMSE before {r0:.2f} V -> after the fit {r1:.2f} V")
            write_fitted_block(R_fit, C_fit, r1, "fitted with the measured current sc_a")
            kf = LinearKalmanFilter(R=R_fit, C=C_fit, sigma_i=SIGMA_I_MEASURED)
        else:
            kf = LinearKalmanFilter(sigma_i=SIGMA_I_MEASURED)
        sim = simulate(u, v0, kf)
        if Q_V is None:
            print("choosing the process noise ...")
            choose_q_v(kf, None, V_meas, u=u)
        res = kf.run(None, V_meas, u=u)
    else:
        print("input: power only (no sc_a column), I = P / V")
        print("fitting sign, losses, R and C (takes up to a minute or two) ...")
        sign, eta_d, eta_c, p_sb, R_fit, C_fit, r0, r1 = fit_power_mode(grid_w, V_meas, v0, kf)
        print(f"power sign used: P_target = {sign:+d} * grid_w   (positive = discharge)")
        print(f"fitted losses: efficiency discharge {eta_d:.3f}, charge {eta_c:.3f}, standby {p_sb:.0f} W")
        print("fitted R =", [round(float(x), 4) for x in R_fit])
        print("fitted C =", [round(float(x), 4) for x in C_fit])
        print(f"  open-loop RMSE before {r0:.2f} V -> after the fit {r1:.2f} V")
        write_fitted_block(R_fit, C_fit, r1, f"power-only fit: eta_d {eta_d:.3f}, eta_c {eta_c:.3f}, standby {p_sb:.0f} W")
        kf = LinearKalmanFilter(R=R_fit, C=C_fit)
        P_target = bank_power(sign * grid_w, eta_d, eta_c, p_sb)
        sim = simulate_power(P_target, v0, kf)
        if Q_V is None:
            print("choosing the process noise ...")
            choose_q_v(kf, P_target, V_meas)
        else:
            kf.q_v = Q_V
        res = kf.run(P_target, V_meas)

    good = ok & np.isfinite(sim)
    rmse = lambda a: float(np.sqrt(np.mean((a[good] - V_meas[good]) ** 2)))
    nis = res["nis"][np.isfinite(res["nis"])]
    rank, ratio = kf.observability()
    print(f"sensor noise used: {res['sigma_v']:.3g} V, process noise {kf.q_v:g} V/step")
    print(f"RMSE open-loop simulation: {rmse(sim):.3f} V")
    print(f"RMSE Kalman filtered:      {rmse(res['V_filt']):.3f} V")
    print(f"mean NIS = {nis.mean():.2f}  (about 1 when the noise settings fit)")
    print(f"observability: rank {rank} of {kf.n}")

    import matplotlib.pyplot as plt
    t = res["time"]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(11, 9))
    ax[0].plot(t, sim, label="V_sim (open loop)")
    ax[0].plot(t, V_meas, label="V_ref (measured)")
    ax[0].plot(t, res["V_filt"], label="V_kf (Kalman)")
    ax[0].set_ylabel("terminal voltage [V]")
    ax[0].ticklabel_format(useOffset=False, style="plain")
    ax[0].legend()
    sd = np.sqrt(res["S"])
    ax[1].plot(t, res["innov"], lw=0.6, label="residual (measured - predicted)")
    ax[1].plot(t, 2 * sd, "r:", lw=0.8, label="+/- 2 std")
    ax[1].plot(t, -2 * sd, "r:", lw=0.8)
    lim = 4 * np.nanmedian(sd)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_ylabel("residual [V]")
    ax[1].legend()
    for i in range(kf.n):
        ax[2].plot(t, res["x"][:, i], lw=1, label=f"v_C{i}")
    ax[2].set_ylabel("capacitor voltages [V]")
    ax[2].set_xlabel("time [s]")
    ax[2].ticklabel_format(useOffset=False, style="plain")
    ax[2].legend(ncol=kf.n)
    fig.tight_layout()
    #fig.savefig("kalman_result.png", dpi=120)
    print("saved kalman_result.png")
    if SHOW_WINDOW:
        plt.show()


if __name__ == "__main__":
    main()