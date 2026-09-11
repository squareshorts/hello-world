from __future__ import annotations

import io
import json
import math
import pathlib

import numpy as np
from remotezip import RemoteZip
from scipy.io import loadmat
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

URL = "https://zenodo.org/api/records/15077957/files/GRASP_S1.zip/content"
SUBJECT = 1
SESSION = 1          # cylindrical grasp
TASK = 2             # 30% MVC
FS_EMG = 2048.0
WIN_S = 0.200
HOP_S = 0.100
CHANNEL_COUNTS = [320, 64, 16, 8, 4]
ALPHAS = np.logspace(-3, 5, 17)
OUT = pathlib.Path("results")
OUT.mkdir(exist_ok=True)


def name(day: int, trial: int) -> str:
    return f"S1/D{day}/S1_Day{day}_Session{SESSION}_Task{TASK}_Trial{trial}.mat"


def load_trial(rz: RemoteZip, day: int, trial: int):
    b = rz.read(name(day, trial))
    d = loadmat(io.BytesIO(b))
    emg = np.asarray(d["data_sEMG"], dtype=np.float64)
    force = np.asarray(d["data_force"], dtype=np.float64).ravel()
    mvc = float(np.asarray(d["MVC"]).ravel()[0])
    return emg, force, mvc


def featurize(emg: np.ndarray, force: np.ndarray):
    win = int(round(WIN_S * FS_EMG))
    hop = int(round(HOP_S * FS_EMG))
    starts = np.arange(0, emg.shape[1] - win + 1, hop, dtype=int)
    centers = (starts + (win - 1) / 2.0) / FS_EMG
    duration = emg.shape[1] / FS_EMG
    # Force and EMG were acquired simultaneously; force record lengths vary slightly.
    # Endpoint normalization avoids accumulating a ~0.1-0.2 s clock-length mismatch.
    ft = np.linspace(0.0, duration, force.size, endpoint=False)
    y = np.interp(centers, ft, force)

    n = len(starts)
    ch = emg.shape[0]
    mav = np.empty((n, ch), dtype=np.float64)
    rms = np.empty_like(mav)
    wl = np.empty_like(mav)
    for i, s in enumerate(starts):
        w = emg[:, s:s + win]
        mav[i] = np.mean(np.abs(w), axis=1)
        rms[i] = np.sqrt(np.mean(w * w, axis=1))
        wl[i] = np.sum(np.abs(np.diff(w, axis=1)), axis=1)
    X = np.concatenate([mav, rms, wl], axis=1)
    return X, y, centers


def rank_channels(X: np.ndarray, y: np.ndarray, n_ch=320):
    mav = X[:, :n_ch]
    yc = y - y.mean()
    yn = np.linalg.norm(yc) + 1e-12
    scores = np.empty(n_ch)
    for j in range(n_ch):
        x = mav[:, j] - mav[:, j].mean()
        scores[j] = abs(float(np.dot(x, yc) / ((np.linalg.norm(x) + 1e-12) * yn)))
    return np.argsort(scores)[::-1], scores


def feature_cols(channels, n_ch=320):
    c = np.asarray(channels, dtype=int)
    return np.concatenate([c, c + n_ch, c + 2 * n_ch])


def fit_model(X, y, channels):
    cols = feature_cols(channels)
    A = X[:, cols]
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (A - mu) / sd
    model = RidgeCV(alphas=ALPHAS, fit_intercept=True)
    model.fit(Z, y)
    return model, mu, sd, cols


def predict(fit, X):
    model, mu, sd, cols = fit
    return model.predict((X[:, cols] - mu) / sd)


def metrics(y, p):
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return {
        "r2": float(r2_score(y, p)),
        "rmse": rmse,
        "nrmse_sd": float(rmse / (np.std(y) + 1e-12)),
        "n": int(len(y)),
    }


def affine_kf(p_cal, y_cal, r_obs, q_gain=1e-5, q_bias=1e-6):
    # Slowly drifting calibration layer: y = gain * base_prediction + bias + noise.
    theta = np.array([1.0, 0.0], dtype=float)
    P = np.diag([0.25, 0.02])
    Q = np.diag([q_gain, q_bias])
    R = max(float(r_obs), 1e-5)
    for p, y in zip(p_cal, y_cal):
        P = P + Q
        H = np.array([float(p), 1.0])
        S = float(H @ P @ H + R)
        K = (P @ H) / S
        theta = theta + K * (float(y) - float(H @ theta))
        P = P - np.outer(K, H) @ P
    return theta


def information_gate(p, t, train_pred_mean, train_pred_sd, threshold=10.0, min_n=20):
    u = (p - train_pred_mean) / (train_pred_sd + 1e-12)
    G = np.zeros((2, 2), dtype=float)
    for i, v in enumerate(u):
        h = np.array([float(v), 1.0])
        G += np.outer(h, h)
        if i + 1 >= min_n:
            lam = float(np.linalg.eigvalsh(G)[0])
            if lam >= threshold:
                return i + 1, float(t[i]), lam
    return len(p), float(t[-1]), float(np.linalg.eigvalsh(G)[0])


def calibrate_and_eval(fit, train_X, train_y, cal, eval_, seconds=None, info=False):
    Xc, yc, tc = cal
    Xe, ye, te = eval_
    pc = predict(fit, Xc)
    pe = predict(fit, Xe)
    ptrain = predict(fit, train_X)
    r_obs = np.var(train_y - ptrain)

    if info:
        ncal, gate_t, lam = information_gate(pc, tc, float(ptrain.mean()), float(ptrain.std()))
    else:
        cutoff = float(seconds)
        idx = np.where(tc <= cutoff)[0]
        ncal = int(len(idx))
        if ncal == 0:
            ncal = 1
        gate_t = float(tc[ncal - 1])
        u = (pc[:ncal] - ptrain.mean()) / (ptrain.std() + 1e-12)
        H = np.column_stack([u, np.ones(ncal)])
        lam = float(np.linalg.eigvalsh(H.T @ H)[0]) if ncal >= 2 else 0.0

    theta = affine_kf(pc[:ncal], yc[:ncal], r_obs)
    pred = theta[0] * pe + theta[1]
    return {
        "metrics": metrics(ye, pred),
        "gain": float(theta[0]),
        "bias": float(theta[1]),
        "n_cal": int(ncal),
        "elapsed_s": float(gate_t),
        "information_lambda_min": float(lam),
    }


def main():
    trials = {}
    with RemoteZip(URL) as rz:
        available = set(rz.namelist())
        needed = [name(d, tr) for d in range(1, 12) for tr in (1, 2)]
        missing = [n for n in needed if n not in available]
        if missing:
            raise RuntimeError(f"Missing expected files: {missing}")

        for day in range(1, 12):
            for tr in (1, 2):
                emg, force, mvc = load_trial(rz, day, tr)
                X, y, t = featurize(emg, force)
                trials[(day, tr)] = (X, y, t, mvc)
                print("LOADED", day, tr, X.shape, float(y.min()), float(y.max()), "MVC", mvc, flush=True)
                del emg, force

    Xtrain, ytrain, ttrain, mvc1 = trials[(1, 1)]
    rank, scores = rank_channels(Xtrain, ytrain)

    result = {
        "dataset": "CEMHSEY GRASP Part I",
        "zenodo_record": 15077957,
        "subject": 1,
        "session": SESSION,
        "grasp": "cylindrical",
        "task": TASK,
        "target_level": "30% MVC",
        "protocol": "Day1 Trial1 training; Day1 Trial2 same-day evaluation; Days2-11 both trials fixed-decoder evaluation; Trial1 calibration -> independent Trial2 evaluation for calibration analyses",
        "features": "MAV + RMS + waveform length; 200-ms window; 100-ms step",
        "force_alignment": "endpoint-normalized interpolation to EMG duration",
        "channel_ranking": "absolute Day1-Trial1 MAV-force correlation",
        "channel_results": {},
        "calibration": {},
    }

    for k in CHANNEL_COUNTS:
        channels = rank[:k]
        fit = fit_model(Xtrain, ytrain, channels)
        same = metrics(trials[(1, 2)][1], predict(fit, trials[(1, 2)][0]))
        per_day = {}
        for day in range(2, 12):
            ys, ps = [], []
            for tr in (1, 2):
                X, y, t, mvc = trials[(day, tr)]
                ys.append(y)
                ps.append(predict(fit, X))
            per_day[str(day)] = metrics(np.concatenate(ys), np.concatenate(ps))
        cross_r2 = [per_day[str(d)]["r2"] for d in range(2, 12)]
        cross_rmse = [per_day[str(d)]["rmse"] for d in range(2, 12)]
        result["channel_results"][str(k)] = {
            "alpha": float(fit[0].alpha_),
            "same_day_trial2": same,
            "per_day": per_day,
            "median_cross_day_r2": float(np.median(cross_r2)),
            "min_cross_day_r2": float(np.min(cross_r2)),
            "max_cross_day_r2": float(np.max(cross_r2)),
            "median_cross_day_rmse": float(np.median(cross_rmse)),
            "selected_channels_1based": [int(x + 1) for x in channels[:min(k, 64)]],
        }
        print("FIXED", k, result["channel_results"][str(k)]["median_cross_day_r2"], flush=True)

    # Calibration is evaluated only on independent Trial2 of each later day.
    for k in (64, 8):
        channels = rank[:k]
        fit = fit_model(Xtrain, ytrain, channels)
        by_day = {}
        for day in range(2, 12):
            cal = trials[(day, 1)][:3]
            eva = trials[(day, 2)][:3]
            base = metrics(eva[1], predict(fit, eva[0]))
            row = {"fixed": base}
            for sec in (5, 10, 15):
                row[f"time_{sec}s"] = calibrate_and_eval(fit, Xtrain, ytrain, cal, eva, seconds=sec)
            row["information_gated"] = calibrate_and_eval(fit, Xtrain, ytrain, cal, eva, info=True)
            by_day[str(day)] = row

        summary = {}
        for method in ("fixed", "time_5s", "time_10s", "time_15s", "information_gated"):
            vals = []
            times = []
            ncal = []
            for day in range(2, 12):
                obj = by_day[str(day)][method]
                if method == "fixed":
                    vals.append(obj["r2"])
                else:
                    vals.append(obj["metrics"]["r2"])
                    times.append(obj["elapsed_s"])
                    ncal.append(obj["n_cal"])
            summary[method] = {
                "median_r2": float(np.median(vals)),
                "min_r2": float(np.min(vals)),
                "max_r2": float(np.max(vals)),
            }
            if times:
                summary[method]["median_elapsed_s"] = float(np.median(times))
                summary[method]["median_n_cal"] = float(np.median(ncal))
        result["calibration"][str(k)] = {"per_day": by_day, "summary": summary}
        print("CAL", k, json.dumps(summary), flush=True)

    out = OUT / "cemhsey_s1_11day_pilot.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULTS_JSON", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
