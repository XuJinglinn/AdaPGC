import os
import subprocess
from datetime import datetime, timedelta
import sys
import shutil
import json


def get_time_prefix():
    time_prefix = (datetime.now() + timedelta(hours=8)).strftime('%Y%m%d_%H%M%S')
    return time_prefix

def init_experiment_dir(log_dir, exp_name):
    time_prefix = get_time_prefix()
    exp_dir = os.path.join(log_dir, f"{time_prefix}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)
    backup_code_dir = os.path.join(exp_dir, 'backup')
    os.makedirs(backup_code_dir, exist_ok=True)
    files = ['output.log', 'result.csv', 'commit_hash.txt', 'remark.md']
    for f in files:
        file_path = os.path.join(exp_dir, f)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as fp:
                pass
    return exp_dir


def add_remark(exp_dir, remark):
    file_path = os.path.join(exp_dir, 'remark.md')
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(remark + '\n')


def save_commit_hash(exp_dir):
    try:
        hash_str = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        hash_str = "Not a git repo"
    with open(os.path.join(exp_dir, 'commit_hash.txt'), 'w') as f:
        f.write(hash_str)

def redirect_output_to_log(log_path):
    logger = Logger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    return logger

def get_result_csv_path(exp_dir):
    return os.path.join(exp_dir, 'result.csv')

def get_backup_code_path(exp_dir):
    return os.path.join(exp_dir, 'backup')

def backup_files(target_dir):
    file_list = ['run_hsic.py', 'TTA/HSIC.py', 'script/run_hsic.sh', 'models/cav_mae.py']
    os.makedirs(target_dir, exist_ok=True)
    for file_path in file_list:
        if os.path.exists(file_path):
            dest_path = os.path.join(target_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
        else:
            print(f"Warning: File does not exist, not copied {file_path}")

def save_args(args, target_dir, filename="args.json"):
    os.makedirs(target_dir, exist_ok=True)
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = dict(args)
    with open(os.path.join(target_dir, filename), "w", encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

