import argparse
import copy
import ctypes
import getpass
import json
import math
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import nullcontext

from guild_utils import cv_util
from guild_utils import mps_controller
from guild_utils import sbatch_template

# import psutil
# import atexit

libc = ctypes.CDLL("libc.so.6")

slurm_template = sbatch_template.slurm_template


def chunk(sequence, chunksize):
    for i in range(0, len(sequence), chunksize):
        yield sequence[i : i + chunksize]


def set_pdeathsig(sig=signal.SIGTERM):
    def callable():
        return libc.prctl(1, sig)

    return callable


def yesno(query, opt_true=("y", "yes"), opt_false=("n", "no")):
    while True:
        answer = input(query + f" [{','.join(opt_true)}|{','.join(opt_false)}]").lower()
        if answer in opt_true:
            return True
        elif answer in opt_false:
            return False


def is_in_singularity():
    if not os.environ.get("SINGULARITY_COMMAND", ""):
        return False
    else:
        return True


def with_singularity(command):
    singularity_container = os.environ["SINGULARITY_CONTAINER"]
    if os.environ["SINGULARITY_BIND"]:
        singularity_bind = "-B " + os.environ["SINGULARITY_BIND"]
    else:
        singularity_bind = ""
    # note: the --nv nvidia extensions warn when nvidia tools are
    # not present, but does not fail
    path = os.environ["PATH"]
    shlex_command = shlex.quote(command)
    cmd = f"singularity exec --env PATH={path} --nv {singularity_bind} {singularity_container} /usr/bin/bash -i -c {shlex_command}"
    return cmd


# #@atexit.register
# def kill_children(sig=signal.SIGINT):
#     print("atexit!")
#     proc = psutil.Process(os.getpid())
#     print(f"children: {proc.children()}")
#     for child in proc.children():
#         pgrp = os.getpgid(child.pid)
#         try:
#             os.killpg(pgrp, sig)
#         except Exception:
#             pass
#         os.kill(child.pid, sig)
#     try:
#         os.killpg(proc.pid, sig)
#     except Exception:
#         pass


class Worker(object):
    def __init__(self, gpu, thread_num, queue, dry_run=False):
        self.gpu = gpu
        self.queue = queue
        self.thread_num = thread_num
        self.dry_run = dry_run

    def do_work(self):
        while True:
            item = self.queue.get()  # timeout=0.01)
            runid = item["id"]
            print(f"Working on {runid}")
            env = copy.deepcopy(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
            try:
                print(f"guild run -y --restart {runid}")
                # subprocess.run(f"guild run -y --restart {runid}", shell=True)
                if not self.dry_run:
                    subprocess.run(
                        shlex.split(f"guild run -y --restart {runid}"),
                        preexec_fn=set_pdeathsig(signal.SIGINT),
                    )
                else:
                    print("(dryrun)")
                print(f"Finished {runid}")
            except Exception:
                traceback.print_exc()
            finally:
                self.queue.task_done()
            time.sleep(0.1)


# filter store and read runs.
class Runs:
    @staticmethod
    def guild_read(runsfilter="-Se"):
        guild_command = f"guild runs --json {runsfilter}"
        print(f"guild_command: {guild_command}")
        output = subprocess.check_output(guild_command, shell=True)
        runs = json.loads(output)
        return runs

    @staticmethod
    def bare_ids_to_json(list_of_ids):
        return [dict(id=runid) for runid in list_of_ids]

    @staticmethod
    def read_json(filename):
        with open(filename, "r") as fobj:
            return json.load(fobj)

    @staticmethod
    def store_json(runs, filename):
        assert runs
        with open(filename, "w") as fobj:
            json.dump(runs, fobj)

    @staticmethod
    def execute(runs, jobs_per_gpu, dry_run=False, use_mps=False):
        # retrieve CUDA Device settings
        CUDA_VISIBLE_DEVICES_str = (
            re.sub(",+", ",", os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", ","))
        ).strip()
        assert CUDA_VISIBLE_DEVICES_str, f"CUDA_VISIBLE_DEVICES does not show devices: {CUDA_VISIBLE_DEVICES_str}"
        CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES_str.split(",")

        run_queue = queue.Queue()
        for run in runs:
            run_queue.put(run)

        with mps_controller.make_mps_controller() if use_mps else nullcontext() as mps:
            if use_mps:
                os.environ.update(mps.get_env_keys())
            workers = [
                Worker(gpu=gpu, thread_num=threadnum, queue=run_queue, dry_run=dry_run)
                for gpu in CUDA_VISIBLE_DEVICES
                for threadnum in range(jobs_per_gpu)
            ]
            threads = [threading.Thread(target=worker.do_work, daemon=True).start() for worker in workers]

            # block until all tasks are done
            time.sleep(3)
            print("-------\n" * 10)
            print(f"PID: {os.getpid()}")
            print("Waiting for queue")
            run_queue.join()
            print("All work completed")
            threads.clear()


def main():
    parser = argparse.ArgumentParser(description="select and schedule guild runs on a slurm cluster.")
    group_runsin = parser.add_mutually_exclusive_group()
    group_runsin.add_argument("--guildfilter", type=str, default=None, help="filter string for guild runs")
    group_runsin.add_argument("--runsfile", type=str, default=None, help="json file result of guild runs")
    group_runsin.add_argument(
        "--runids",
        nargs="+",
    )

    parser.add_argument("--store-runs", type=str, default=None, help="filename to write filtered runs to")

    group_execute = parser.add_mutually_exclusive_group()
    group_execute.add_argument("--sbatch", action="store_true")
    parser.add_argument("--sbatch-yes", action="store_true")
    parser.add_argument("--sbatch-verbose", action="store_true")
    parser.add_argument(
        "--convert-cuda-visible-uuids", action="store_true", help="Use nvidia-smi to convert visible devices to uuids."
    )
    parser.add_argument("--use-mps", action="store_true", help="Should an nvidia-cuda-mps-control daemon be launched?")
    group_execute.add_argument("--exec", action="store_true")

    parser.add_argument("--jobs-per-gpu", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--partition", type=str, default="IFIgpu")
    parser.add_argument("--exclude-nodes", type=str, default="headnode")
    parser.add_argument("--guild-home", type=str, default=None, help="GUILD_HOME directory")

    # sbatch additional parameters
    parser.add_argument("--jobname", type=str, default="guild-runner")
    parser.add_argument("--nice", type=int, default=0)
    parser.add_argument(
        "--use-nodes", type=int, default=-1, help="how many parallel sbatch files and thus nodes to use"
    )
    parser.add_argument("--num-gpus", type=int, default=4, help="How many GPUs to request via slurm. Minimum is 1.")
    parser.add_argument("--num-cpus", type=int, default=27, help="How many CPUs per job.")

    args = parser.parse_args()

    if args.use_mps and not args.convert_cuda_visible_uuids:
        print("NOTE: '--use-mps' implies  '--convert-cuda-visible-uuids'")
        args.convert_cuda_visible_uuids = True

    if args.guild_home:
        os.environ["GUILD_HOME"] = args.guild_home
        guild_home = f"export GUILD_HOME='{args.guild_home}'"
    else:
        guild_home = ""

    # ---- read runs...
    runs = None
    if args.guildfilter:
        runs = Runs.guild_read(args.guildfilter)
    elif args.runids:
        runs = Runs.bare_ids_to_json(args.runids)
    elif args.runsfile:
        runs = Runs.read_json(args.runsfile)

    if args.store_runs:
        Runs.store_json(runs, args.store_runs)
    # ----

    # sbatch or execute:
    if args.exec:
        if args.convert_cuda_visible_uuids:
            cv_util.convert_cuda_visible_devices(os.environ)
        print("execute runs!")
        Runs.execute(runs, args.jobs_per_gpu, dry_run=args.dry_run, use_mps=args.use_mps)
    elif args.sbatch:
        print("should create sbatch...")

        nr_of_runs = len(runs)
        worker_slots_per_node = args.num_gpus * args.jobs_per_gpu
        frac_num_nodes = nr_of_runs / worker_slots_per_node
        full_nodes = int(math.ceil(frac_num_nodes))
        nr_of_nodes = args.use_nodes
        print(f"num jobs: {nr_of_runs}")
        print(f"slots per gpu: {args.jobs_per_gpu}")
        print(f"requested cpus: {args.num_cpus}")
        if args.use_nodes > full_nodes:
            raise RuntimeError(
                (
                    f"We have {nr_of_runs} runs, {args.jobs_per_gpu} jobs/GPU, {args.num_gpus} GPUs, "
                    f"and thus {worker_slots_per_node} slots per node. "
                    f"{args.use_nodes} are requested, but we can fill only {frac_num_nodes} "
                    f"({full_nodes}) nodes. Use less jobs per node or less nodes."
                )
            )
        if nr_of_nodes < 1:
            over_count = int(abs(args.use_nodes))
            nr_of_nodes = int(math.ceil((nr_of_runs / worker_slots_per_node) / over_count))
            print(
                (
                    f"Automatic node calculation used, every node should execute {over_count}"
                    f" jobs sequentially, thus: {nr_of_nodes} nodes"
                )
            )

        # todo chunk heer
        nr_of_jobs_per_node = int(math.ceil(nr_of_runs / nr_of_nodes))
        joblens = ", ".join([str(len(chunk_runs)) for i, chunk_runs in enumerate(chunk(runs, nr_of_jobs_per_node))])
        print(f"jobs per node: [ {joblens} ]")

        if not args.sbatch_yes:
            if not yesno("Continue?"):
                sys.exit(-1)
        for i, chunk_runs in enumerate(chunk(runs, nr_of_jobs_per_node)):
            chunk_runids = [run["id"] for run in chunk_runs]
            flags_passthrough = []
            if args.use_mps:
                flags_passthrough.append("--use-mps")
            if args.convert_cuda_visible_uuids:
                flags_passthrough.append("--convert-cuda-visible-uuids")
            flags_passthrough_string = " " + " ".join(flags_passthrough) + " "
            command = f"{sys.executable} {__file__} --exec {flags_passthrough_string} --runids {' '.join(chunk_runids)} --jobs-per-gpu {args.jobs_per_gpu} --num-gpus {args.num_gpus} --num-cpus {args.num_cpus}"
            if is_in_singularity():
                command = with_singularity(command)
            slurm_content = slurm_template.substitute(
                user=getpass.getuser(),
                cmd=command,
                jobname=f"{args.jobname}-{i}",
                guild_home=guild_home,
                num_gpus=args.num_gpus,
                # num_cores=27,
                num_cores=args.num_cpus,
                partition=args.partition,
                exclude_nodes=args.exclude_nodes,
            )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as sbash:
                sbash.write(slurm_content)
                sbash.flush()
                command = f"sbatch --nice={args.nice} {sbash.name} "
                if args.sbatch_verbose:
                    print(f"\n--- sbatch file for job {i}, {sbash.name} ---\n")
                    subprocess.run(f"cat {sbash.name}", shell=True)
                print(f"command: {command}")
                if not args.dry_run:
                    subprocess.run(command, shell=True)
                pass  # create batch with runs here.
        print(f"\n=== {i+1} job files===\n")


if __name__ == "__main__":
    main()
