import argparse
import subprocess
import sys
import tempfile

import joblib
import pandas as pd
import tqdm


def create_trial_args(args):
    operation = None
    for arg in args:
        if "=" not in arg and not arg.startswith("-"):
            operation = arg
            break
    print(f"Detected operation: {operation}")

    with tempfile.NamedTemporaryFile() as ntf:
        command = ["guild", "run", f"--save-trials={ntf.name}"] + args
        # command = ["guild", "run"] + args
        command = " ".join(command)
        print(repr(command))
        subprocess.check_call(command, shell=True)
        result = pd.read_csv(ntf.name)
    trial_args = [row.dropna().to_dict() for i, row in result.iterrows()]
    return operation, trial_args


def trial_args2trial_commands(operation, trial_args):
    trial_commands = []
    for trial_arg in trial_args:
        flags = " ".join([f"{k}={v}" for k, v in trial_arg.items()])
        trial_commands.append(f"guild run --yes {operation} --stage {flags}")
    return trial_commands


def parallel_stage_trials(trial_commands, n_jobs=None):
    if n_jobs is None:
        n_jobs = joblib.cpu_count()

    def execute_command(command):
        subprocess.check_call(command, shell=True)

    jobs = [joblib.delayed(execute_command)(command) for command in trial_commands]
    result = joblib.Parallel(n_jobs=n_jobs)(tqdm.tqdm(jobs))
    return result


def split_list(the_list, the_element, other=None):
    if other is None:
        other = []
    try:
        i = the_list.index(the_element)
        other.append(the_list[:i])
        return split_list(the_list[i + 1 :], the_element, other)
    except ValueError:
        other.append(the_list)
    return other


def main():
    # split arguments into "--" before after.

    split_args = split_list(sys.argv[1:], "--")
    guild_run_args = None
    stager_args = None
    if len(split_args) == 1:
        (guild_run_args,) = split_args
    elif len(split_args) == 2:
        guild_run_args, stager_args = split_args
    elif len(split_args) == 3:
        _, guild_run_args, stager_args = split_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--n-jobs", type=int, default=None)
    pargs = parser.parse_args(stager_args)

    operation, trial_args = create_trial_args(guild_run_args)
    trial_commands = trial_args2trial_commands(operation, trial_args)
    print("\n".join(trial_commands))
    if not pargs.dry_run:
        parallel_stage_trials(trial_commands, n_jobs=pargs.n_jobs)


if __name__ == "__main__":
    main()
