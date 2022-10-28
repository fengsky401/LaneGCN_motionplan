import os
process_num = 10
pre_file = "/data2/queenie/av1_data/centerline/test/centerline_speed_40000_81000.pkl"#"/data2/queenie/LaneGCN_motionplan/data_av1/train/centerline_speed_140000_180000.pkl"
#process_dir = "/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/train/data"
save_dir = "/data2/queenie/av1_data/plan/test"#"/data/queenie/plan"#"/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/save_train_frenet/test_limit"
command_file_name = "gen_test_40000_81000.sh"
keyword = "test_gen_40000_81000"
#filenum_list = [int(i.split(".")[0]) for i in os.listdir(process_dir)]
#file_num = len(filenum_list)
#max_filenum = max(filenum_list)
#min_filenum = min(filenum_list)
each_increment = (41000-0)//process_num
#import torchaudio
#torchaudio.save()
with open(command_file_name,"w") as f:

    for i in range(process_num):
        start_num = each_increment*i + 40000
        end_num = each_increment*(i+1) + 40000
        command_line = "nohup python gen_feasible_path_debug.py --pre_file %s --save_dir %s  --start_num %d  --end_num  %d >a_%s_%d.out 2>&1 & \n" \
                       %(pre_file,save_dir,start_num,end_num,keyword,i)
        #command_line = "nohup python gen_feasible_path_debug.py --pre_file ./data_av1/centerline_speed/centerline_speed_100000_150000.pkl --start_num %d --end_num %d >a_%s_%d.out 2>&1 & \n" %(start_num,end_num,keyword,i) 
        f.writelines(command_line)




