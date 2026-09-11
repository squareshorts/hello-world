import json
import os
import pathlib

import cemhsey_longitudinal as m

s = int(os.environ["SUBJECT"])
m.SUBJECT = s
m.URL = f"https://zenodo.org/api/records/15077957/files/GRASP_S{s}.zip/content"
m.name = lambda day, trial: f"S{s}/D{day}/S{s}_Day{day}_Session{m.SESSION}_Task{m.TASK}_Trial{trial}.mat"

m.main()

src = pathlib.Path("results/cemhsey_s1_11day_pilot.json")
obj = json.loads(src.read_text(encoding="utf-8"))
obj["subject"] = s
obj["archive"] = f"GRASP_S{s}.zip"
dst = pathlib.Path(f"results/cemhsey_s{s}_11day_pilot.json")
dst.write_text(json.dumps(obj, indent=2), encoding="utf-8")
if src != dst:
    src.unlink(missing_ok=True)
print("WRAPPER_RESULT", s, json.dumps({k: v["median_cross_day_r2"] for k, v in obj["channel_results"].items()}), flush=True)
