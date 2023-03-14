import os
import sys
import subprocess
from glob import glob
import argparse



if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-r", "--root_dir")
    argParser.add_argument("-p","--pattern")
    argParser.add_argument("-l","--launch",action='store_true')

    args = argParser.parse_args()

    pattern = args.pattern if args.pattern is not None else '*'
    files_list = glob(f'{args.root_dir}{pattern}')
    config_list = [file.split('/')[-1] for file in files_list]

    if args.launch:
        for config in config_list:
            subprocess.run(["python","core/Main.py","--config_path",f"{args.root_dir}{config}"])
    else:
        print(config_list)

