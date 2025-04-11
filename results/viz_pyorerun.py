import numpy as np
import pickle
from plotly.colors import DEFAULT_PLOTLY_COLORS
from pyorerun import PhaseRerun, BiorbdModel
from pyorerun.multi_frame_rate_phase_rerun import MultiFrameRatePhaseRerun

folder = "backflip_Vpost_submission_collision_feb25"
filename = "htc/sol_1_CVG.pkl"
model_path = "../models/Model2D_7Dof_2C_5M_CL_V3.bioMod"

# remove "rgb(" and ")" and split by ","
colors = [color[4:-1].split(",") for color in DEFAULT_PLOTLY_COLORS]
colors = [(int(color[0]), int(color[1]), int(color[2])) for color in colors]
colors = colors + colors + colors  # duplicate the colors to have enough for all the phases

# Charger les données
phase_reruns = []
data = pickle.load(open(folder + f"/{filename}", "rb"))
time = np.concatenate([np.array(time) for time in data["time"]], axis=0).squeeze()
q = np.concatenate([np.array(q) for q in data["q"]], axis=1)

phase_reruns.append(PhaseRerun(t_span=time, window=f"simulation"))
m = BiorbdModel(model_path)
m.options.mesh_color = colors[0]
phase_reruns[-1].add_animated_model(m, q)

mrr2 = MultiFrameRatePhaseRerun(phase_reruns=phase_reruns)
mrr2.rerun_by_frame("all_multistart")
