import subprocess
import time
suffix_list = ["_R12_0"+str(i)+"_N_22" for i in [0]+list(range(5,10))]

for suffix in suffix_list:
    configname = 'config.py'
    with open(configname,'w') as f:
        f.write("# This is a lazy man's config file - don't do this at home, kids!\n")
        f.write("experiment_id = '20161107_0'\n")
        f.write("suffix = '"+suffix+"'\n")
        # flush the config file buffers before calling the actual processing
        # from https://stackoverflow.com/a/9824894/3791466
        f.flush()
    time.sleep(1)

    subprocess.call(["python","analyse.py"])
