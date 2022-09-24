torch_file = "/mnt/sda/queenie2/LaneGCN_motionplan/data_av1/centerline_speed/centerline_speed_50000_100000.pt"
pickle_file = "/mnt/sda/queenie2/LaneGCN_motionplan/data_av1/centerline_speed/centerline_speed_50000_100000.pkl"

import torch
import pickle

ct = torch.load(torch_file)
with open(pickle_file,'wb') as f:
    pickle.dump(ct,f)
