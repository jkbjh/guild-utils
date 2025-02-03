# guild_utils

This repository contains utilities to use [guild](https://guild.ai/) on a [slurm](https://slurm.schedmd.com/) based
cluster.  It needs to be installed in the same virtual environment as
guild (and the experiment code).

The main workflow is to stage all the guild experiments using
```
guild run  --stage ...
```

or, in case of a batched operation such as a grid search:
```
guild run --stage-trials ...
```

The utils also contain the `guild-parallel-stager`, which internally uses
```
guild run --save-trials path_to.csv ...
```
to generate runs, but creates the runs independently (not a main batch operation) as staged runs in *parallel* (and thus, on our cluster, a lot faster).

Example usage:
```
guild-parallel-stager my-guild-operation my-parameter=[value1,value2,value3] ...
```

The staged runs can then be scheduled on the slurm cluster using the `guild-slurm-runner`:

```
$ guild-slurm-runner --help
usage: guild-slurm-runner [-h]
                          [--guildfilter GUILDFILTER | --runsfile RUNSFILE | --runids RUNIDS [RUNIDS ...]]
                          [--store-runs STORE_RUNS] [--sbatch] [--sbatch-yes]
                          [--sbatch-verbose] [--convert-cuda-visible-uuids]
                          [--use-mps] [--exec]
                          [--workers-per-job WORKERS_PER_JOB] [--dry-run]
                          [--partition PARTITION]
                          [--exclude-nodes EXCLUDE_NODES]
                          [--guild-home GUILD_HOME]
                          [--create-template CREATE_TEMPLATE]
                          [--template-file TEMPLATE_FILE] [--list-templates]
                          [--jobname JOBNAME] [--nice NICE]
                          [--use-jobs USE_JOBS] [--num-gpus NUM_GPUS]
                          [--num-cpus NUM_CPUS]

select and schedule guild runs on a slurm cluster.

optional arguments:
  -h, --help            show this help message and exit
  --guildfilter GUILDFILTER
                        filter string for guild runs (default: None)
  --runsfile RUNSFILE   json file result of guild runs (default: None)
  --runids RUNIDS [RUNIDS ...]
  --store-runs STORE_RUNS
                        filename to write filtered runs to (default: None)
  --sbatch
  --sbatch-yes
  --sbatch-verbose
  --convert-cuda-visible-uuids
                        Use nvidia-smi to convert visible devices to uuids.
                        (default: False)
  --use-mps             Should an nvidia-cuda-mps-control daemon be launched?
                        (default: False)
  --exec
  --workers-per-job WORKERS_PER_JOB
                        how many workers per slurm job. These will be
                        distributed evenly across GPUs or vice versa.
                        (default: 5)
  --dry-run
  --partition PARTITION
  --exclude-nodes EXCLUDE_NODES
  --guild-home GUILD_HOME
                        GUILD_HOME directory (default: None)
  --create-template CREATE_TEMPLATE
                        Create a template (choose template from a list)
                        (default: None)
  --template-file TEMPLATE_FILE
                        Path to sbatch (string.Template) template (default:
                        ~/.guild_utils_sbatch_template)
  --list-templates
  --jobname JOBNAME
  --nice NICE
  --use-jobs USE_JOBS   how many parallel sbatch files and thus jobs to use
                        (default: -1)
  --num-gpus NUM_GPUS   How many GPUs to request via slurm. Minimum is 1.
                        (default: 4)
  --num-cpus NUM_CPUS   How many CPUs per job. (default: 27)
```
