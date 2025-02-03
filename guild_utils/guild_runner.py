import argparse
import copy
import ctypes
import grp
import itertools
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
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from fractions import Fraction

from guild_utils import cv_util
from guild_utils import mps_controller
from guild_utils.sbatch_template import SlurmTemplate

# import psutil
# import atexit

libc = ctypes.CDLL("libc.so.6")


def float_to_fraction(number):
    MAX_DENOMINATOR = int(1e4)
    if isinstance(number, Fraction):
        return number
    return Fraction(number).limit_denominator(MAX_DENOMINATOR)


def is_sequence_but_not_string(obj):
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def deprecated_argument(type_converter, argument_name=""):
    def deprecated_argument(value):
        warnings.warn(f"The argument {argument_name} is deprecated.", DeprecationWarning)
        return type_converter(value)

    return deprecated_argument


def chunk(sequence, chunksize):
    for i in range(0, len(sequence), chunksize):
        yield sequence[i : i + chunksize]


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def create_worker_kwargs(cuda_visible_devices, workers_per_gpu_ratio, number_of_subjob_workers=None, **kwargs):
    workers_per_gpu_ratio = float_to_fraction(workers_per_gpu_ratio)
    if number_of_subjob_workers is None:
        number_of_subjob_workers = len(cuda_visible_devices) * workers_per_gpu_ratio
    number_of_subjob_workers = float_to_fraction(number_of_subjob_workers)
    assert (
        number_of_subjob_workers.denominator == 1
    ), f"Subjob workers must be integer, but is {number_of_subjob_workers}"
    number_of_subjob_workers = number_of_subjob_workers.numerator
    worker_threadnums = list(range(number_of_subjob_workers))

    if len(cuda_visible_devices) > 0:
        grouped_gpus = list(
            map(list, grouper(cuda_visible_devices, workers_per_gpu_ratio.denominator)),
        )
        grouped_workers = list(grouper(worker_threadnums, workers_per_gpu_ratio.numerator))
        worker_gpu_pairs = list(zip(grouped_workers, grouped_gpus))

        worker_kwargs = []
        for workers, gpus in worker_gpu_pairs:
            for worker, gpus in list(zip(workers, [gpus] * len(workers))):
                if len(gpus) == 1:
                    (gpus,) = gpus
                worker_kwargs.append(dict(gpus=gpus, thread_num=worker, **kwargs))
    else:
        worker_kwargs = []
        for worker in worker_threadnums:
            worker_kwargs.append(dict(gpus=[], thread_num=worker, **kwargs))
    return worker_kwargs


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


def get_user_and_group():
    user = os.getlogin()
    group = grp.getgrgid(os.getgid()).gr_name
    return user, group


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
    def __init__(self, gpus, thread_num, queue, dry_run=False):
        if is_sequence_but_not_string(gpus):
            gpus = ",".join(gpus)
        self.gpus = gpus
        self.queue = queue
        self.thread_num = thread_num
        self.dry_run = dry_run

    def do_work(self):
        while True:
            item = self.queue.get()  # timeout=0.01)
            runid = item["id"]
            print(f"Working on {runid}")
            env = copy.deepcopy(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = self.gpus
            try:
                print(f"guild run -y --restart {runid}")
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
    def execute(runs, number_of_subjob_workers=None, dry_run=False, use_mps=False, number_of_gpus=None):
        """
        This function executes a given list of runs.
        """
        workers_per_gpu = number_of_subjob_workers / max(number_of_gpus, 1)  # even if there are zero gpus,
        # set ratio to 1.

        # retrieve CUDA Device settings
        CUDA_VISIBLE_DEVICES_str = (
            re.sub(",+", ",", os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", ","))
        ).strip()
        if number_of_gpus > 0:
            assert CUDA_VISIBLE_DEVICES_str, f"CUDA_VISIBLE_DEVICES does not show devices: {CUDA_VISIBLE_DEVICES_str}"
        cuda_visible_devices = CUDA_VISIBLE_DEVICES_str.split(",")
        if number_of_gpus is not None:
            if len(cuda_visible_devices) >= number_of_gpus:
                raise RuntimeError(f"expected {number_of_gpus} required, but {len(cuda_visible_devices)} available")

        run_queue = queue.Queue()
        for run in runs:
            run_queue.put(run)

        with mps_controller.make_mps_controller() if use_mps else nullcontext() as mps:
            if use_mps:
                os.environ.update(mps.get_env_keys())

            worker_kwargs = create_worker_kwargs(
                cuda_visible_devices=cuda_visible_devices,
                workers_per_gpu=workers_per_gpu,
                number_of_subjob_workers=number_of_subjob_workers,
                queue=run_queue,
                dry_run=dry_run,
            )
            workers = [Worker(**kwargs) for kwargs in worker_kwargs]
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
    parser = argparse.ArgumentParser(
        description="select and schedule guild runs on a slurm cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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

    # parser.add_argument("--jobs-per-gpu", type=int, default=5)
    parser.add_argument(
        "--workers-per-job",
        type=int,
        default=5,
        help=("how many workers per slurm job. " "These will be distributed evenly across GPUs or vice versa."),
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--partition", type=str, default="IFIgpu")
    parser.add_argument("--exclude-nodes", type=str, default="headnode")
    parser.add_argument("--guild-home", type=str, default=None, help="GUILD_HOME directory")

    # sbatch additional parameters
    parser.add_argument(
        "--create-template", type=str, required=False, help="Create a template (choose template from a list)"
    )
    parser.add_argument(
        "--template-file",
        type=str,
        default=SlurmTemplate.get_default_template_filename(),
        help="Path to sbatch (string.Template) template",
    )
    parser.add_argument("--list-templates", action="store_true")
    parser.add_argument("--jobname", type=str, default="guild-runner")
    parser.add_argument("--nice", type=int, default=0)
    parser.add_argument("--use-jobs", type=int, default=-1, help="how many parallel sbatch files and thus jobs to use")
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

    # sbatch templates
    if args.list_templates:
        SlurmTemplate.print_defaults()
        parser.exit("")

    if args.create_template:
        SlurmTemplate.create_template(args.template_file, args.create_template)
        SlurmTemplate.print_template(args.create_template)
    slurm_template = SlurmTemplate(args.template_file)

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
        Runs.execute(
            runs, args.workers_per_job, dry_run=args.dry_run, use_mps=args.use_mps, number_of_gpus=args.num_gpus
        )
    elif args.sbatch:
        print("should create sbatch...")

        nr_of_runs = len(runs)
        worker_slots_per_slurmjob = args.workers_per_job
        frac_num_jobs = nr_of_runs / worker_slots_per_slurmjob
        full_jobs = int(math.ceil(frac_num_jobs))
        nr_of_jobs = args.use_jobs
        print(f"num jobs: {nr_of_runs}")
        print(f"workers per slurmjob: {worker_slots_per_slurmjob}")
        if args.num_gpus > 0:
            if args.num_gpus > worker_slots_per_slurmjob:
                print(f"gpus per worker: {float_to_fraction(args.num_gpus / worker_slots_per_slurmjob)}")
                print("Using Multiple-GPUs")
                if args.num_gpus % worker_slots_per_slurmjob != 0:
                    raise RuntimeError(
                        f"Requesting {args.num_gpus} GPUs, and {worker_slots_per_slurmjob} subjobs:"
                        " GPUs cannot be distributed evenly among subjobs."
                    )
            else:
                print(f"workers per gpu: {float_to_fraction(worker_slots_per_slurmjob / args.num_gpus)}")
                print("Running parallel jobs on GPU.")
        print(f"requested cpus: {args.num_cpus}")
        if args.use_jobs > full_jobs:
            raise RuntimeError(
                (
                    f"We have {nr_of_runs} runs, {args.num_gpus} GPUs, "
                    f"and {worker_slots_per_slurmjob} workers/slots per slurm job. "
                    f"{args.use_jobs} are requested, but we can fill only {frac_num_jobs} "
                    f"({full_jobs}) slurm jobs. Use less workers per slurm-job or less slurm-jobs."
                )
            )
        if nr_of_jobs < 1:
            over_count = int(abs(args.use_jobs))
            nr_of_jobs = int(math.ceil((nr_of_runs / worker_slots_per_slurmjob) / over_count))
            print(
                (
                    f"Automatic node calculation used, every node should execute {over_count}"
                    f" jobs sequentially, thus: {nr_of_jobs} jobs"
                )
            )

        # Chunking
        nr_of_jobs_per_node = int(math.ceil(nr_of_runs / nr_of_jobs))
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
            command = f"{sys.executable} {__file__} --exec {flags_passthrough_string} --runids {' '.join(chunk_runids)} --workers-per-job {args.workers_per_job} --num-gpus {args.num_gpus} --num-cpus {args.num_cpus}"
            if is_in_singularity():
                command = with_singularity(command)
            username, groupname = get_user_and_group()
            slurm_content = slurm_template.substitute(
                user=username,
                cmd=command,
                jobname=f"{args.jobname}-{i}",
                guild_home=guild_home,
                num_gpus=args.num_gpus,
                num_cores=args.num_cpus,
                partition=args.partition,
                exclude_nodes=args.exclude_nodes,
                group=groupname,
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
