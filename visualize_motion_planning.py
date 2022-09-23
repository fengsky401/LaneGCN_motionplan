from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import numpy as np
import torch
from pykalman import KalmanFilter
from argoverse.map_representation.map_api import ArgoverseMap
from utils.others import process_input, process_xy, sample_speed, process_xy_back, get_direction, prediction,generate_offset,convert_to_frenet,close_list_index,cal_dist,cal_paral,cal_speed_list,inc_length
from utils.frenet_optimal_trajectory import generate_target_course,frenet_optimal_planning,calc_frenet_paths
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from tqdm import tqdm
import time
import os

import numpy as np

OBS_LEN=20
# Only support nearby in our implementation
def get_nearby_lane_feature_ls(am, agent_df, obs_len, city_name, lane_radius):

    query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
    nearby_lane_ids = am.get_lane_ids_in_xy_bbox(
        query_x, query_y, city_name, lane_radius)

    return nearby_lane_ids

def draw_map_path(traj_df,path_data,avm,plot_dir,path_name):
    city_name_,data_ = process_input(traj_df)
    agent_df = None

    for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
        # sorted already according to timestamp
        if obj_type == 'AGENT':
            agent_df = remain_df
            start_x, start_y = agent_df[['X', 'Y']].values[0]
            agent_x_end, agent_y_end = agent_df[['X', 'Y']].values[-1]
            query_x, query_y = agent_df[['X', 'Y']].values[OBS_LEN-1]
            norm_center = np.array([query_x, query_y])
            break
        else:
            raise ValueError(f"cannot find 'agent' object type")


    #Draw map & centerline
    nearby_lane_ids = get_nearby_lane_feature_ls(avm,agent_df,20,city_name_,lane_radius=65)
    nearby_lane_ids = [[lane_id] for lane_id in nearby_lane_ids]
    #candidate_centerlines = avm.get_candidate_centerlines_for_traj(data_[0:19], city_name_, viz=False,max_search_radius = 65.0)
    candidate_centerlines = avm.get_cl_from_lane_seq(nearby_lane_ids, city_name_)
    candidate_centerlines_list = [np.expand_dims(candidate_centerlines_single,axis=0) for candidate_centerlines_single in candidate_centerlines]
    lane_centerlines = np.concatenate(candidate_centerlines_list,axis=0)
    plt.figure(0, figsize=(8, 7))
    x_min = np.min(lane_centerlines[:, :, 0])
    x_max = np.max(lane_centerlines[:, :, 0])
    y_min = np.min(lane_centerlines[:, :, 1])
    y_max = np.max(lane_centerlines[:, :, 1])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    for lane_cl in lane_centerlines:
        plt.plot(
            lane_cl[:,0],
            lane_cl[:,1],
            "--",
            color="grey",
            alpha=1,
            linewidth=1,
            zorder=0,
            label="centerline"
        )

    #Draw target history & Future
    plt.plot(data_[:20,0],data_[:20,1],"--",color="orange",alpha=1,linewidth=1,zorder=1,label="target history")
    plt.plot(data_[20:,0],data_[20:,1],"--",color="red",alpha=1,linewidth=1,zorder=1,label="target future trajectory")

    #Draw feasible path
    for feasible_traj in path_data['generate_traj']:
        plt.plot(feasible_traj[:,0],
                 feasible_traj[:,1],
                 "--",
                 color="blue",
                 alpha=1,
                 linewidth=1,
                 zorder=0,
                 label="feasible future trajectory")


    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    plt.axis("off")

    plt.savefig(os.path.join(plot_dir,path_name+".png"))
    plt.close()
    #print("city name:",city_name_)

if __name__ == "__main__":
    root_dir = '/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/train/data'
    save_dir = '/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/feasible_traj'
    plot_dir = "/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/path_plot/train7_seperate_step"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    #name=os.listdir(root_dir)
    avm = ArgoverseMap()
    Afl = ArgoverseForecastingLoader
    afl = Afl(root_dir)
    path_file_list = [name.split(".")[0] for name in os.listdir(save_dir)]
    for path_name in path_file_list:
        map_file_name = os.path.join(root_dir,path_name+".csv")
        path_file_name = os.path.join(save_dir,path_name+".path")
        scene_info = afl.get(map_file_name)
        path_info = torch.load(path_file_name)
        draw_map_path(scene_info.seq_df,path_info,avm,plot_dir,path_name)