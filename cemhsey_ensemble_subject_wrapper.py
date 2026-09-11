import json
import os
import pathlib

import cemhsey_longitudinal as m
import cemhsey_sparse_ensemble as ens

s = int(os.environ["SUBJECT"])
m.SUBJECT = s
m.URL = f"https://zenodo.org/api/records/15077957/files/GRASP_S{s}.zip/content"
m.name = lambda day, trial: f"S{s}/D{day}/S{s}_Day{day}_Session{m.SESSION}_Task{m.TASK}_Trial{trial}.mat"

ens.main()

src = pathlib.Path("results/cemhsey_s1_sparse_ensemble.json")
obj = json.loads(src.read_text(encoding="utf-8"))
obj["subject"] = s
obj["archive"] = f"GRASP_S{s}.zip"
dst = pathlib.Path(f"results/cemhsey_s{s}_sparse_ensemble.json")
dst.write_text(json.dumps(obj, indent=2), encoding="utf-8")
if src != dst:
    src.unlink(missing_ok=True)

compact = {
    k: {method: vals["summary"][method]["median_r2"] for method in vals["summary"]}
    for k, vals in obj["sets"].items()
}
print("ENSEMBLE_WRAPPER_RESULT", s, json.dumps(compact), flush=True)
