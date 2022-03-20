import argparse
import getpass
import threading

import json5 as json
import paramiko

run_num = 1
run_lock = threading.Lock()
max_num_runs = 10


def repeat_machine(host, username, password, command, workdir, pypath, device=""):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=username, password=password)

    global run_num
    global run_lock
    global max_num_runs

    while run_num <= max_num_runs:
        with run_lock:
            this_run_num = run_num
            run_num += 1

        post_str = "--post" + " " + str(this_run_num)

        print("running " + str(this_run_num) + " on " + host + " " + device)
        stdin, stdout, stderr = client.exec_command(
            "cd " + workdir + "; PYTHONPATH=" + pypath + " " + command + " " + post_str,
            get_pty=True)

        try:
            exit_status = stdout.channel.recv_exit_status()
        except KeyboardInterrupt as e:
            client.close()
            raise e

        # print(stdout.read().decode("utf-8"))
        print(stderr.read().decode("utf-8"))
        print("Finished " + str(this_run_num))
        if exit_status != 0:
            print("Error: Exit status " + str(exit_status))
            break


def run():
    p = argparse.ArgumentParser()
    global run_num
    global max_num_runs

    p.add_argument("config")
    p.add_argument("--run-start", type=int, default=1)
    p.add_argument("--num-runs", type=int, default=10)
    p.add_argument("--username")
    p.add_argument("--password")

    a = p.parse_args()
    config_file = a.config
    run_num = a.run_start
    max_num_runs = a.num_runs

    with open(config_file) as f:
        config = json.load(f)

    path = config["python"] + " " + config["script"]
    args = config["args"]
    command = path + " " + args

    password = getpass.getpass()  # NOTE: This needs to be run in debug mode if in pycharm

    for machine in config["machines"]:
        devarg = "--device " + str(machine["device"])
        threading.Thread(target=repeat_machine, kwargs={
            "host": machine["host"],
            "password": password,
            "command": command + " " + devarg,
            "username": a.username,
            "pypath": config["pythonpath"],
            "workdir": config["workdir"],
            "device": str(machine["device"])
        }).start()


if __name__ == '__main__':
    run()
