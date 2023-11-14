import functools
import os
import string

DEFAULT_TEMPLATES = {
    "uibk_ifi_cluster": """
#!/bin/bash -l
#SBATCH --partition=${partition}
#SBATCH --exclude=${exclude_nodes}
#SBATCH --job-name=$jobname
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${user}@uibk.ac.at
#SBATCH --account=${group} # change to your group, if not default unix group
#SBATCH -n ${num_cores} # number of cores
#SBATCH --gres gpu:${num_gpus} # number of gpus
#SBATCH -o /scratch/${user}/slurm_logs/slurm.%N.%j.out # STDOUT
#SBATCH -e /scratch/${user}/slurm_logs/slurm.%N.%j.err # STDERR

printenv
$guild_home

$cmd

wait
"""
}


class SlurmTemplate:
    def __init__(self, filename):
        filename = os.path.expanduser(filename)
        if not os.path.exists(filename):
            raise OSError(f"Sbatch template '{filename}' does not exist!")
        self.filename = filename
        with open(filename, "rt") as fobj:
            self.template_string = fobj.read().strip()  # strip trailing/leading
        self.template = string.Template(self.template_string)

    @functools.wraps(string.Template.substitute)
    def substitute(self, *args, **kwargs):
        return self.template.substitute(*args, **kwargs)

    @staticmethod
    def get_default_template_filename():
        return "~/.guild_utils_sbatch_template"

    @staticmethod
    def list_defaults():
        return list(sorted(DEFAULT_TEMPLATES.keys()))

    @staticmethod
    def print_defaults():
        print("Available templates:")
        print("\n".join(DEFAULT_TEMPLATES.keys()))

    @staticmethod
    def print_template(templatename):
        print("Available templates:")
        print("Template: \n----------")
        print(DEFAULT_TEMPLATES[templatename])
        print("----------")

    @classmethod
    def create_template(cls, filename, templatename):
        if templatename not in DEFAULT_TEMPLATES:
            raise OSError(f"template '{templatename}', does not exist. Templates: {cls.list_defaults()}")
        filename = os.path.expanduser(filename)

        if os.path.exists(filename):
            raise OSError(f"File '{filename}' exists. Cowardly refusing to overwrite template file.")
        with open(filename, "wt") as fobj:
            fobj.write(DEFAULT_TEMPLATES[templatename])
