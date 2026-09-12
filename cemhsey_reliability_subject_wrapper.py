import os
import cemhsey_longitudinal as m
import cemhsey_reliability_kf as rkf

s = int(os.environ["SUBJECT"])
m.SUBJECT = s
m.URL = f"https://zenodo.org/api/records/15077957/files/GRASP_S{s}.zip/content"
m.name = lambda day, trial: f"S{s}/D{day}/S{s}_Day{day}_Session{m.SESSION}_Task{m.TASK}_Trial{trial}.mat"
rkf.main()
