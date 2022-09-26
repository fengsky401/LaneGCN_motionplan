import os
<<<<<<< HEAD
process_num = 10
process_dir = "/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/train/data"
save_dir = "/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/save_train_frenet/test_limit"
command_file_name = "gen_train_100000_150000.sh"
keyword = "train_gen_100000_150000"
=======
process_num = 12
process_dir = "/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/train/data"
save_dir = "/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/save_train_frenet/test_limit"
command_file_name = "gen_train_50000_100000.sh"
keyword = "train_gen_50000_100000"
>>>>>>> e6fdb08cc9e309ec43287335e5fc5c4c0d2be491
#filenum_list = [int(i.split(".")[0]) for i in os.listdir(process_dir)]
#file_num = len(filenum_list)
#max_filenum = max(filenum_list)
#min_filenum = min(filenum_list)
<<<<<<< HEAD
each_increment = (150000-100000)//process_num
=======
each_increment = (50000-0)//process_num
>>>>>>> e6fdb08cc9e309ec43287335e5fc5c4c0d2be491
#import torchaudio
#torchaudio.save()
with open(command_file_name,"w") as f:

    for i in range(process_num):
<<<<<<< HEAD
        start_num = each_increment*i + 100000
        end_num = each_increment*(i+1) + 100000
        #command_line = "nohup python planning_n_args.py --root_dir %s --save_dir %s  --start_num %d  --end_num  %d >a_%s_%d.out 2>&1 & \n" \
        #               %(process_dir,save_dir,start_num,end_num,keyword,i)
        command_line = "nohup python gen_feasible_path_debug.py --pre_file ./data_av1/centerline_speed/centerline_speed_100000_150000.pkl --start_num %d --end_num %d >a_%s_%d.out 2>&1 & \n" %(start_num,end_num,keyword,i) 
=======
        start_num = each_increment*i + 50000
        end_num = each_increment*(i+1) + 50000
        #command_line = "nohup python planning_n_args.py --root_dir %s --save_dir %s  --start_num %d  --end_num  %d >a_%s_%d.out 2>&1 & \n" \
        #               %(process_dir,save_dir,start_num,end_num,keyword,i)
        command_line = "nohup python gen_feasible_path_debug.py --pre_file ./data_av1/centerline_speed/centerline_speed_50000_100000.pkl  --start_num %d --end_num %d >a_%s_%d.out 2>&1 & \n" %(start_num,end_num,keyword,i) 
>>>>>>> e6fdb08cc9e309ec43287335e5fc5c4c0d2be491
        f.writelines(command_line)




