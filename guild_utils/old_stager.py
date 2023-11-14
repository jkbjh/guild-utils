import warnings

from .parallel_stager import main as stager_main


def main(*args, **kwargs):
    warnings.warn("guild-utils-stager called, which is deprecated, " "use guild-parallel-stager instead.")
    return stager_main(*args, **kwargs)
