import subprocess
import os
def rtrace(inputf,ouputf):
    #print(path,file)
    cur_dir = os.getcwd()
    # os.chdir("./rivuletpy")
    os.chdir(r"../graph_projtct/rivuletpy")
    cmd=["python rtrace","-f",inputf,"-t 0 -o",ouputf]
    cmd=' '.join(cmd)
    print(cmd)
    subprocess.call(cmd, shell=True)
    os.chdir(cur_dir)
    print(cmd)
    # return 'success'
# test("2","1")
# name = 'CaiWenHe'
# rtrace('../../graph_data/new_AV_data_origin/A/' + name + '.nii.gz', '../../graph_data/new_AV_data_origin/center_line/' + name + '_A.txt')