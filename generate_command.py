import os
process_num = 12
process_dir = "/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/train/data"
save_dir = "/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/save_train_frenet/test_limit"
command_file_name = "gen_train_50000_100000.sh"
keyword = "train_gen_50000_100000"
#filenum_list = [int(i.split(".")[0]) for i in os.listdir(process_dir)]
#file_num = len(filenum_list)
#max_filenum = max(filenum_list)
#min_filenum = min(filenum_list)
each_increment = (50000-0)//process_num
#import torchaudio
#torchaudio.save()
with open(command_file_name,"w") as f:

    for i in range(process_num):
        start_num = each_increment*i + 50000
        end_num = each_increment*(i+1) + 50000
        #command_line = "nohup python planning_n_args.py --root_dir %s --save_dir %s  --start_num %d  --end_num  %d >a_%s_%d.out 2>&1 & \n" \
        #               %(process_dir,save_dir,start_num,end_num,keyword,i)
        command_line = "nohup python gen_feasible_path_debug.py --pre_file ./data_av1/centerline_speed/centerline_speed_50000_100000.pkl  --start_num %d --end_num %d >a_%s_%d.out 2>&1 & \n" %(start_num,end_num,keyword,i) 
        f.writelines(command_line)




