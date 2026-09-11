from __future__ import annotations

import json
import numpy as np
from remotezip import RemoteZip

import cemhsey_longitudinal as m
import cemhsey_sparse_ensemble as ens


def metrics(y, p):
    return m.metrics(y, p)


def kf_fuse(P, r0, q, alpha, init_x=None, init_p=None):
    """Scalar latent-force KF with channel-specific adaptive observation variance.

    P has shape channels x time and contains independent single-channel force estimates.
    Reliability uses only contemporaneous decoder disagreement; no force labels are used.
    """
    k, n = P.shape
    out = np.empty(n, dtype=float)
    x = float(np.median(P[:, 0]) if init_x is None else init_x)
    pvar = float(np.var(P[:, 0]) + np.median(r0) if init_p is None else init_p)
    eps = 1e-10
    r0 = np.maximum(np.asarray(r0, dtype=float), eps)
    for t in range(n):
        # random-walk prediction
        pvar += q
        med = float(np.median(P[:, t]))
        mad = float(np.median(np.abs(P[:, t] - med)) + 1e-8)
        disagreement = np.abs(P[:, t] - med) / (1.4826 * mad + 1e-8)
        rt = r0 * np.exp(np.minimum(alpha * disagreement**2, 12.0))
        # information-form multi-observation update for H_i = 1
        prior_prec = 1.0 / max(pvar, eps)
        obs_prec = 1.0 / rt
        post_prec = prior_prec + float(np.sum(obs_prec))
        x = (prior_prec * x + float(np.sum(obs_prec * P[:, t]))) / post_prec
        pvar = 1.0 / post_prec
        out[t] = x
    return out


def static_precision_fuse(P, r0):
    w = 1.0 / np.maximum(r0, 1e-10)
    w /= np.sum(w)
    return np.sum(w[:, None] * P, axis=0)


def main():
    trials = {}
    with RemoteZip(m.URL) as rz:
        for day in range(1, 12):
            for tr in (1, 2):
                emg, force, mvc = m.load_trial(rz, day, tr)
                X, y, t = m.featurize(emg, force)
                trials[(day, tr)] = (X, y, t, mvc)
                print('LOADED', day, tr, flush=True)

    Xtr, ytr, _, _ = trials[(1, 1)]
    Xval, yval, _, _ = trials[(1, 2)]
    rank, _ = m.rank_channels(Xtr, ytr)
    out = {
        'subject': int(getattr(m, 'SUBJECT', 1)),
        'protocol': 'single-channel models fit on Day1 Trial1; KF hyperparameters selected on Day1 Trial2 only; Days2-11 untouched; reliability uses prediction disagreement only',
        'sets': {}
    }

    for k in (8, 16):
        ch = rank[:k]
        models = ens.train_single_channel_models(Xtr, ytr, ch)
        Pv = ens.prediction_matrix(models, Xval)
        # Baseline channel observation variances from Day1 validation residuals.
        r0 = np.mean((Pv - yval[None, :])**2, axis=1) + 1e-8
        # Random-walk process scale anchored to Day1 force increments.
        dy_var = float(np.var(np.diff(ytr)) + 1e-8)

        grid = []
        for q_scale in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            q = q_scale * dy_var
            for alpha in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
                pred = kf_fuse(Pv, r0, q, alpha)
                mm = metrics(yval, pred)
                grid.append((mm['rmse'], -mm['r2'], q_scale, alpha, mm))
        grid.sort(key=lambda z: (z[0], z[1]))
        _, _, q_scale, alpha, val_best = grid[0]
        q = q_scale * dy_var

        static_val = metrics(yval, static_precision_fuse(Pv, r0))
        median_val = metrics(yval, np.median(Pv, axis=0))
        print('KF_SELECT', k, json.dumps({'q_scale': q_scale, 'alpha': alpha, 'validation': val_best, 'static_validation': static_val, 'median_validation': median_val}), flush=True)

        per_day = {}
        for day in range(2, 12):
            ys, Ps = [], []
            for tr in (1, 2):
                X, y, _, _ = trials[(day, tr)]
                ys.append(y)
                Ps.append(ens.prediction_matrix(models, X))
            y = np.concatenate(ys)
            P = np.concatenate(Ps, axis=1)
            pred_kf = kf_fuse(P, r0, q, alpha)
            pred_static = static_precision_fuse(P, r0)
            pred_median = np.median(P, axis=0)
            pred_consensus, retention = ens.robust_consensus(P)
            per_day[str(day)] = {
                'reliability_kf': metrics(y, pred_kf),
                'static_precision': metrics(y, pred_static),
                'median': metrics(y, pred_median),
                'consensus_3mad': metrics(y, pred_consensus),
                'retention': [float(v) for v in retention],
            }

        summary = {}
        for method in ('reliability_kf', 'static_precision', 'median', 'consensus_3mad'):
            vals = [per_day[str(d)][method]['r2'] for d in range(2, 12)]
            summary[method] = {
                'median_r2': float(np.median(vals)),
                'min_r2': float(np.min(vals)),
                'max_r2': float(np.max(vals)),
            }
        out['sets'][str(k)] = {
            'channels_1based': [int(v + 1) for v in ch],
            'r0': [float(v) for v in r0],
            'q_scale': float(q_scale),
            'alpha': float(alpha),
            'day1_validation': val_best,
            'per_day': per_day,
            'summary': summary,
        }
        print('KF_SUMMARY', k, json.dumps(summary), flush=True)

    p = m.OUT / f"cemhsey_s{int(getattr(m, 'SUBJECT', 1))}_reliability_kf.json"
    p.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('KF_JSON', json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
