<<<<<<< HEAD
torch_file = "./data_av1/centerline_speed/centerline_speed_100000_150000.pt"
pickle_file = "./data_av1/centerline_speed/centerline_speed_100000_150000.pkl"
=======
torch_file = "/mnt/sda/queenie2/LaneGCN_motionplan/data_av1/centerline_speed/centerline_speed_50000_100000.pt"
pickle_file = "/mnt/sda/queenie2/LaneGCN_motionplan/data_av1/centerline_speed/centerline_speed_50000_100000.pkl"
>>>>>>> e6fdb08cc9e309ec43287335e5fc5c4c0d2be491

import torch
import pickle

ct = torch.load(torch_file)
with open(pickle_file,'wb') as f:
    pickle.dump(ct,f)
