torch_file = "/data2/queenie/av1_data/centerline/test/centerline_speed_40000_81000.pt"#"/data2/queenie/LaneGCN_motionplan/data_av1/train/centerline_speed_180000_220000.pt"#"./data_av1/centerline_speed/centerline_speed_100000_150000.pt"
pickle_file = "/data2/queenie/av1_data/centerline/test/centerline_speed_40000_81000.pkl"#"/data2/queenie/LaneGCN_motionplan/data_av1/train/centerline_speed_180000_220000.pkl"#"./data_av1/centerline_speed/centerline_speed_100000_150000.pkl"

import torch
import pickle

ct = torch.load(torch_file)
with open(pickle_file,'wb') as f:
    pickle.dump(ct,f)
