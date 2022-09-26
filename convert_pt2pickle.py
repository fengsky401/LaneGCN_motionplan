torch_file = "./data_av1/centerline_speed/centerline_speed_100000_150000.pt"
pickle_file = "./data_av1/centerline_speed/centerline_speed_100000_150000.pkl"

import torch
import pickle

ct = torch.load(torch_file)
with open(pickle_file,'wb') as f:
    pickle.dump(ct,f)
