# guild_utils

This repository contains utilities to use guild on a slurm based
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

The utils also contain the `guild-utils-stager`, which internally uses
```
guild run --save-trials path_to.csv ...
```
to generate runs, but creates the runs independently (not a main batch operation) as staged runs in *parallel* (and thus, on our cluster, a lot faster).

Example usage:
```
guild-utils-stager my-guild-operation my-parameter=[value1,value2,value3] ...
```

The staged runs can then be scheduled on the slurm cluster using the `guild-slurm-runner`:

```
$ guild-slurm-runner --help
usage: guild-slurm-runner [-h] [--guildfilter GUILDFILTER | --runsfile RUNSFILE | --runids RUNIDS [RUNIDS ...]] [--store-runs STORE_RUNS] [--sbatch] [--sbatch-yes] [--sbatch-verbose] [--exec] [--jobs-per-gpu JOBS_PER_GPU] [--dry-run]
                          [--partition PARTITION] [--exclude-nodes EXCLUDE_NODES] [--guild-home GUILD_HOME] [--jobname JOBNAME] [--nice NICE] [--use-nodes USE_NODES] [--num-gpus NUM_GPUS] [--num-cpus NUM_CPUS]

select and schedule guild runs on a slurm cluster.

options:
  -h, --help            show this help message and exit
  --guildfilter GUILDFILTER
                        filter string for guild runs
  --runsfile RUNSFILE   json file result of guild runs
  --runids RUNIDS [RUNIDS ...]
  --store-runs STORE_RUNS
                        filename to write filtered runs to
  --sbatch
  --sbatch-yes
  --sbatch-verbose
  --exec
  --jobs-per-gpu JOBS_PER_GPU
  --dry-run
  --partition PARTITION
  --exclude-nodes EXCLUDE_NODES
  --guild-home GUILD_HOME
                        GUILD_HOME directory
  --jobname JOBNAME
  --nice NICE
  --use-nodes USE_NODES
                        how many parallel sbatch files and thus nodes to use
  --num-gpus NUM_GPUS   How many GPUs to request via slumr. Minimum is 1.
  --num-cpus NUM_CPUS   How many CPUs per job.
```
