import subprocess
suffix_list = ["_R12_0"+str(i)+"_N_22" for i in range(6,10)]

for suffix in suffix_list:
    configname = 'config.py'
    with open(configname,'w') as f:
        f.write("# This is a lazy man's config file - don't do this at home, kids!\n")
        f.write("experiment_id = '20161107_0'\n")
        f.write("suffix = '"+suffix+"'\n")

    subprocess.call(["python","sample.py"])
