import string


class SlurmTemplate:
    def __init__(self, filename):
        self.filename = filename
        with open(filename, "rt") as fobj:
            self.template_string = fobj.read().strip()  # strip trailing/leading
        self.template = string.Template(self.template_string)


slurm_template = string.Template(
    """
#!/bin/bash -l
#SBATCH --partition=${partition}
#SBATCH --exclude=${exclude_nodes}
#SBATCH --job-name=$jobname
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${user}@uibk.ac.at
#SBATCH --account=iis ##change to your group
#SBATCH -n ${num_cores} # number of cores
#SBATCH --gres gpu:${num_gpus} # number of gpus
#SBATCH -o /scratch/${user}/slurm_logs/slurm.%N.%j.out # STDOUT
#SBATCH -e /scratch/${user}/slurm_logs/slurm.%N.%j.err # STDERR

printenv
$guild_home

$cmd

wait
""".strip()  # trailing/leading
)
