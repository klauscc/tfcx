# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2019/09/24
#   description:
#
#================================================================

import subprocess


def print_log(message, log_file):
    """ print log and save to log_file

    Args:
        message: The message want to print.
        log_file: The file to save the log.

    """
    print(message)
    with open(log_file, "at") as f:
        f.write(str(message) + "\n")


def get_vacant_gpu():
    com = "nvidia-smi|sed -n '/%/p'|sed 's/|/\\n/g'|sed -n '/MiB/p'|sed 's/ //g'|sed 's/MiB/\\n/'|sed '/\\//d'"
    gpum = subprocess.check_output(com, shell=True)
    gpum = gpum.decode('utf-8').split('\n')
    gpum = gpum[:-1]
    for i, d in enumerate(gpum):
        gpum[i] = int(gpum[i])
    gpu_id = gpum.index(min(gpum))
    if len(gpum) == 4:
        gpu_id = 3 - gpu_id
    return gpu_id
