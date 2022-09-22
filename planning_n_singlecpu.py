from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap
import os
import warnings
import torch
import argparse

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from tqdm import tqdm
import time

ACCELERATE_SEARCH = 1
SEARCH_SPEED = 10.0




if __name__ == "__main__":
    def parse_args():

        parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
        parser.add_argument('--root_dir', type=str, default='/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/train/data')
        parser.add_argument('--save_dir', type=str, default='/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/save_train_frenet/test_cpu')
        parser.add_argument('--start_num', type=int, default=0)
        parser.add_argument('--end_num', type=int, default=100)

        args = parser.parse_args()

        return args

    args = parse_args()
    root_dir = args.root_dir#'/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/train/data'
    save_dir = args.save_dir#'/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/save_train_frenet/test'
    start_num = args.start_num
    end_num = args.end_num
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    name=os.listdir(root_dir)
    Afl = ArgoverseForecastingLoader
    avm = ArgoverseMap()

    afl = Afl(root_dir)
    print('Total number of sequences:', len(afl))
    info_dict = []
    check_list = ["52", "39", "60"]
    for recur in range(1000000):
        for path_name_ext in tqdm(afl.seq_list):
            single_num = path_name_ext.parts[-1].split(".")[0]
            if (int(single_num) > start_num) and (int(single_num) < end_num):
                #afl_ = afl.get(path_name_ext)
                path, name_ext = os.path.split(path_name_ext)
                name, ext = os.path.splitext(name_ext)
                time_begin = time.time()
                # save_centerline, generate_traj = model_generator(avm, afl_.seq_df, 20)
                # data_dict = {"save_centerline": save_centerline, "generate_traj": generate_traj}
                data_dict = {}
                torch.save(data_dict, os.path.join(save_dir, name + ".path"))
                time_cost = time.time() - time_begin
                print("time cost per scene:", time_cost)

