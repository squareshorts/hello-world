from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import urllib.request
from dataclasses import dataclass

import numpy as np
import wfdb
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "_emg_force_data"
OUT = ROOT / "results"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

BASE = "https://physionet.org/files/hd-semg/2.0.0/ndof_dataset"
COMBOS = [1, 8, 15]
SESSIONS = [1, 2]
SAMPLES = [1, 2]
SUBJECT = "01"
CHANNEL_COUNTS = [256, 64, 16, 8, 4]
ALPHAS = np.logspace(-3, 4, 16)


def download(url: str, path: pathlib.Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"DOWNLOAD {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def get_record(session: int, combo: int, sample: int, kind: str):
    folder = DATA / f"subject{SUBJECT}_session{session}"
    folder.mkdir(exist_ok=True)
    stem = f"ndof_{kind}_combination{combo}_sample{sample}"
    for ext in ("dat", "hea"):
        url = f"{BASE}/subject{SUBJECT}_session{session}/{stem}.{ext}"
        download(url, folder / f"{stem}.{ext}")
    rec = wfdb.rdrecord(str(folder / stem))
    return np.asarray(rec.p_signal, dtype=np.float64), float(rec.fs)


@dataclass
class Trial:
    session: int
    combo: int
    sample: int
    X: np.ndarray
    y: np.ndarray
    times: np.ndarray


def feature_trial(session: int, combo: int, sample: int) -> Trial:
    emg, fs_e = get_record(session, combo, sample, "preprocess")
    force, fs_f = get_record(session, combo, sample, "force")
    emg = np.nan_to_num(emg, copy=False)
    force = np.nan_to_num(force, copy=False)
    if emg.ndim != 2 or emg.shape[1] != 256:
        raise RuntimeError(f"Unexpected EMG shape: {emg.shape}")
    if force.ndim == 1:
        force = force[:, None]

    # Hyser benchmark-compatible temporal resolution: 200 ms, 50% overlap.
    win = int(round(0.200 * fs_e))
    hop = int(round(0.100 * fs_e))
    starts = np.arange(0, emg.shape[0] - win + 1, hop, dtype=int)
    centers = (starts + (win - 1) / 2.0) / fs_e

    mav = np.empty((len(starts), emg.shape[1]), dtype=np.float64)
    rms = np.empty_like(mav)
    wl = np.empty_like(mav)
    for i, s in enumerate(starts):
        w = emg[s:s + win]
        mav[i] = np.mean(np.abs(w), axis=0)
        rms[i] = np.sqrt(np.mean(w * w, axis=0))
        wl[i] = np.sum(np.abs(np.diff(w, axis=0)), axis=0)
    X = np.concatenate([mav, rms, wl], axis=1)

    ft = np.arange(force.shape[0], dtype=np.float64) / fs_f
    # Scalar grip-force analogue: Euclidean magnitude across the five finger-force channels.
    force_mag = np.linalg.norm(force, axis=1)
    y = np.interp(centers, ft, force_mag)
    return Trial(session, combo, sample, X, y, centers)


def concat_trials(trials):
    return np.vstack([t.X for t in trials]), np.concatenate([t.y for t in trials])


def rank_channels(X: np.ndarray, y: np.ndarray, n_ch: int = 256) -> np.ndarray:
    mav = X[:, :n_ch]
    yc = y - np.mean(y)
    ys = np.sqrt(np.sum(yc * yc)) + 1e-12
    scores = np.zeros(n_ch)
    for j in range(n_ch):
        xc = mav[:, j] - np.mean(mav[:, j])
        scores[j] = abs(float(np.dot(xc, yc) / ((np.sqrt(np.sum(xc * xc)) + 1e-12) * ys)))
    return np.argsort(scores)[::-1]


def feature_cols(channels: np.ndarray, n_ch: int = 256) -> np.ndarray:
    return np.concatenate([channels, channels + n_ch, channels + 2 * n_ch])


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def metrics(y, p):
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    sd = float(np.std(y, ddof=0))
    rng = float(np.max(y) - np.min(y))
    return {
        "r2": float(r2_score(y, p)),
        "rmse": rmse,
        "nrmse_sd": rmse / sd if sd > 0 else math.nan,
        "nrmse_range": rmse / rng if rng > 0 else math.nan,
        "n": int(len(y)),
    }


def fit_ridge(train_trials, channels):
    X, y = concat_trials(train_trials)
    cols = feature_cols(np.asarray(channels, dtype=int))
    X = X[:, cols]
    mu, sd = standardize_fit(X)
    model = RidgeCV(alphas=ALPHAS, fit_intercept=True)
    model.fit((X - mu) / sd, y)
    return model, mu, sd, cols


def predict_trials(model_tuple, trials):
    model, mu, sd, cols = model_tuple
    preds = []
    ys = []
    per_trial = []
    for t in trials:
        p = model.predict((t.X[:, cols] - mu) / sd)
        preds.append(p)
        ys.append(t.y)
        per_trial.append((t, p))
    return np.concatenate(ys), np.concatenate(preds), per_trial


def estimate_kalman_params(y_train, z_train):
    mu = float(np.mean(y_train))
    a = y_train[:-1] - mu
    b = y_train[1:] - mu
    phi = float(np.dot(a, b) / (np.dot(a, a) + 1e-12))
    phi = float(np.clip(phi, 0.0, 0.9999))
    q = float(np.var(b - phi * a)) + 1e-12
    r = float(np.var(y_train - z_train)) + 1e-12
    return mu, phi, q, r


def kalman_filter_obs(z, params):
    mu, phi, q, r = params
    x = float(z[0])
    P = max(q, 1e-9)
    out = np.empty_like(z)
    out[0] = x
    for i in range(1, len(z)):
        xp = mu + phi * (x - mu)
        Pp = phi * phi * P + q
        K = Pp / (Pp + r)
        x = xp + K * (float(z[i]) - xp)
        P = (1.0 - K) * Pp
        out[i] = x
    return out


def kalman_predict(model_tuple, train_trials, test_trials):
    yt, zt, _ = predict_trials(model_tuple, train_trials)
    kp = estimate_kalman_params(yt, zt)
    ys = []
    ps = []
    _, _, per = predict_trials(model_tuple, test_trials)
    for t, z in per:
        ys.append(t.y)
        ps.append(kalman_filter_obs(z, kp))
    return np.concatenate(ys), np.concatenate(ps), kp


def affine_calibration(model_tuple, cal_trials, eval_trials, seconds):
    if seconds <= 0:
        y, p, _ = predict_trials(model_tuple, eval_trials)
        return metrics(y, p), {"slope": 1.0, "intercept": 0.0, "n_cal": 0}
    cy = []
    cp = []
    remaining = []
    for t in cal_trials:
        _, _, per = predict_trials(model_tuple, [t])
        pred = per[0][1]
        mask = t.times <= (t.times[0] + seconds)
        if np.count_nonzero(mask) < 3:
            continue
        cy.append(t.y[mask])
        cp.append(pred[mask])
        # Remaining part of sample 1 is evaluated, not reused for calibration.
        idx = np.where(~mask)[0]
        if len(idx):
            remaining.append((t.y[idx], pred[idx]))
    cy = np.concatenate(cy)
    cp = np.concatenate(cp)
    A = np.column_stack([cp, np.ones_like(cp)])
    slope, intercept = np.linalg.lstsq(A, cy, rcond=None)[0]
    ys = [v[0] for v in remaining]
    ps = [slope * v[1] + intercept for v in remaining]
    y2, p2, _ = predict_trials(model_tuple, eval_trials)
    ys.append(y2)
    ps.append(slope * p2 + intercept)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return metrics(y, p), {"slope": float(slope), "intercept": float(intercept), "n_cal": int(len(cy))}


def main():
    trials = []
    for session in SESSIONS:
        for combo in COMBOS:
            for sample in SAMPLES:
                t = feature_trial(session, combo, sample)
                print("TRIAL", session, combo, sample, t.X.shape, t.y.shape,
                      float(np.min(t.y)), float(np.max(t.y)), flush=True)
                trials.append(t)

    train = [t for t in trials if t.session == 1 and t.sample == 1]
    within = [t for t in trials if t.session == 1 and t.sample == 2]
    cross1 = [t for t in trials if t.session == 2 and t.sample == 1]
    cross2 = [t for t in trials if t.session == 2 and t.sample == 2]
    cross = cross1 + cross2
    Xtr, ytr = concat_trials(train)
    rank = rank_channels(Xtr, ytr)

    results = {
        "dataset": "Hyser v2.0.0 / N-DoF",
        "subject": SUBJECT,
        "combinations": COMBOS,
        "design": "session1/sample1 training; session1/sample2 within-day test; both session2 samples cross-day test",
        "features": "MAV + RMS + waveform length, 200-ms windows, 100-ms step",
        "target": "Euclidean magnitude of the five finger-force channels",
        "channel_selection": "training-only absolute MAV-force correlation ranking",
        "channel_results": {},
    }

    for k in CHANNEL_COUNTS:
        channels = rank[:k]
        fit = fit_ridge(train, channels)
        yw, pw, _ = predict_trials(fit, within)
        yc, pc, _ = predict_trials(fit, cross)
        ykw, pkw, kp = kalman_predict(fit, train, within)
        ykc, pkc, _ = kalman_predict(fit, train, cross)
        results["channel_results"][str(k)] = {
            "ridge_alpha": float(fit[0].alpha_),
            "selected_channels_1based": [int(x + 1) for x in channels],
            "within_day_ridge": metrics(yw, pw),
            "cross_day_ridge": metrics(yc, pc),
            "within_day_kalman": metrics(ykw, pkw),
            "cross_day_kalman": metrics(ykc, pkc),
            "kalman": {"mean": float(kp[0]), "phi": float(kp[1]), "q": float(kp[2]), "r": float(kp[3])},
        }
        print("RESULT", k, results["channel_results"][str(k)], flush=True)

    # Small labeled calibration budget on the 8-channel deployment candidate.
    fit8 = fit_ridge(train, rank[:8])
    calibration = {}
    for sec in [0, 2, 5, 10]:
        if sec == 0:
            y0, p0, _ = predict_trials(fit8, cross)
            calibration[str(sec)] = {"metrics": metrics(y0, p0), "fit": {"slope": 1.0, "intercept": 0.0, "n_cal": 0}}
        else:
            m, pars = affine_calibration(fit8, cross1, cross2, sec)
            calibration[str(sec)] = {"metrics": m, "fit": pars}
    results["eight_channel_affine_calibration_seconds"] = calibration

    # Quantify the basic cross-day penalty as delta R2 for each channel count.
    results["cross_day_penalty_delta_r2"] = {
        k: results["channel_results"][k]["cross_day_ridge"]["r2"] - results["channel_results"][k]["within_day_ridge"]["r2"]
        for k in results["channel_results"]
    }

    out = OUT / "hyser_subject01_pilot.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== RESULTS_JSON ===")
    print(out.read_text(encoding="utf-8"))
    print("=== END_RESULTS_JSON ===")


if __name__ == "__main__":
    main()
