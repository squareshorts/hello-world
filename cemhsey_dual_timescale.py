from __future__ import annotations

import json
import os
import numpy as np
from remotezip import RemoteZip

import cemhsey_longitudinal as m
import cemhsey_sparse_ensemble as ens
import cemhsey_reliability_kf as rkf


def robust_loc_scale(x):
    x = np.asarray(x, dtype=float)
    loc = float(np.median(x))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = float(max(q75 - q25, 1e-5))
    return loc, scale


def align_predictions(P, cal_fused, ref_fused):
    """Target-free session alignment from unlabeled Trial1 to Day1 prediction distribution."""
    c0, cs = robust_loc_scale(cal_fused)
    r0, rs = robust_loc_scale(ref_fused)
    gain = rs / cs
    return r0 + gain * (P - c0), float(gain), float(r0 - gain * c0)


def fuse_3mad(P):
    return ens.robust_consensus(P)[0]


def main():
    s = int(os.environ.get('SUBJECT', getattr(m, 'SUBJECT', 1)))
    m.SUBJECT = s
    m.URL = f"https://zenodo.org/api/records/15077957/files/GRASP_S{s}.zip/content"
    m.name = lambda day, trial: f"S{s}/D{day}/S{s}_Day{day}_Session{m.SESSION}_Task{m.TASK}_Trial{trial}.mat"

    trials = {}
    with RemoteZip(m.URL) as rz:
        for day in range(1, 12):
            for tr in (1, 2):
                emg, force, mvc = m.load_trial(rz, day, tr)
                X, y, t = m.featurize(emg, force)
                trials[(day, tr)] = (X, y, t, mvc)
                print('LOADED', s, day, tr, flush=True)

    Xtr, ytr, _, _ = trials[(1, 1)]
    Xval, yval, _, _ = trials[(1, 2)]
    rank, _ = m.rank_channels(Xtr, ytr)
    out = {
        'subject': s,
        'protocol': 'Day1 Trial1 training; Day1 Trial2 validation/reference; later-day Trial1 unlabeled session alignment; later-day Trial2 independent evaluation; no later-day force labels used for adaptation',
        'sets': {}
    }

    for k in (8, 16):
        ch = rank[:k]
        models = ens.train_single_channel_models(Xtr, ytr, ch)
        Pv = ens.prediction_matrix(models, Xval)
        ref_cons = fuse_3mad(Pv)
        r0 = np.mean((Pv - yval[None, :])**2, axis=1) + 1e-8
        dy_var = float(np.var(np.diff(ytr)) + 1e-8)

        # KF settings selected on Day1 Trial2 only.
        grid = []
        for q_scale in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            q = q_scale * dy_var
            for alpha in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
                pred = rkf.kf_fuse(Pv, r0, q, alpha)
                mm = m.metrics(yval, pred)
                grid.append((mm['rmse'], -mm['r2'], q_scale, alpha))
        grid.sort()
        _, _, q_scale, alpha = grid[0]
        q = q_scale * dy_var

        per_day = {}
        for day in range(2, 12):
            Xc, yc, _, _ = trials[(day, 1)]
            Xe, ye, _, _ = trials[(day, 2)]
            Pc = ens.prediction_matrix(models, Xc)
            Pe = ens.prediction_matrix(models, Xe)
            cal_cons = fuse_3mad(Pc)

            # Session-level robust affine alignment is estimated entirely from predictions.
            Pe_aligned, gain, bias = align_predictions(Pe, cal_cons, ref_cons)
            unaligned_cons = fuse_3mad(Pe)
            aligned_cons = fuse_3mad(Pe_aligned)
            aligned_kf = rkf.kf_fuse(Pe_aligned, r0, q, alpha)
            per_day[str(day)] = {
                'unaligned_consensus': m.metrics(ye, unaligned_cons),
                'session_aligned_consensus': m.metrics(ye, aligned_cons),
                'dual_timescale_kf': m.metrics(ye, aligned_kf),
                'alignment_gain': gain,
                'alignment_bias': bias,
            }

        summary = {}
        for method in ('unaligned_consensus', 'session_aligned_consensus', 'dual_timescale_kf'):
            vals = [per_day[str(d)][method]['r2'] for d in range(2, 12)]
            summary[method] = {
                'median_r2': float(np.median(vals)),
                'min_r2': float(np.min(vals)),
                'max_r2': float(np.max(vals)),
                'mean_r2': float(np.mean(vals)),
            }
        out['sets'][str(k)] = {
            'channels_1based': [int(v + 1) for v in ch],
            'q_scale': float(q_scale),
            'alpha': float(alpha),
            'per_day': per_day,
            'summary': summary,
        }
        print('DUAL_SUMMARY', s, k, json.dumps(summary), flush=True)

    p = m.OUT / f'cemhsey_s{s}_dual_timescale.json'
    p.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('DUAL_RESULT', s, json.dumps({k:v['summary'] for k,v in out['sets'].items()}), flush=True)


if __name__ == '__main__':
    main()
