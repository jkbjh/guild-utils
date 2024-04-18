import guild.ipy as guild
import numpy as np
import pandas as pd
import tqdm


def load_runs():
    df = guild.runs()
    dff = df.guild_flags()
    runs = df.join(dff, lsuffix="short")
    runs.loc[:, "runtime"] = [
        max(runline.runshort.run.get("stopped", np.nan) - runline.runshort.run.get("started", 0), 0) * 1e-6
        for idx, runline in runs.iterrows()
    ]
    return runs


def load_scalars(runs):
    all_scalars_ = [row[1].scalars() for row in tqdm.tqdm(runs.iterrows(), total=len(runs))]
    all_scalars = pd.concat(all_scalars_)
    pivot_scalars_ = all_scalars.drop(columns="prefix").pivot_table(
        values=set(all_scalars.columns) - set(["run", "tag", "prefix"]), columns="tag", index="run"
    )
    pivot_scalars = pd.DataFrame(pivot_scalars_)
    pivot_scalars.columns = [f"{b}/{a}" for a, b in pivot_scalars_.columns]
    all_data = runs.set_index("run").join(pivot_scalars)
    return all_data


def get_scalar_detail(row):
    if row.status != "completed":
        return None
    scalars = row.scalars_detail()
    unique_paths = scalars.path.unique()
    if len(unique_paths):
        last_path = unique_paths[-1]  # noqa: F841
        scalars = scalars.query("path == @last_path")
    scalars.loc[:, "run"] = [run.id for run in scalars["run"]]
    return scalars.set_index("run")


def load_scalars_detailed(runs, eval_tags, skip_step=4):
    """
    eval_tags example: eval_tags= set(["eval/mean_ep_length", "eval/mean_reward"])
    """
    detailed_all_scalars_ = [get_scalar_detail(row[1]) for row in tqdm.tqdm(runs.iterrows(), total=len(runs))]
    the_big = pd.concat(detailed_all_scalars_)
    good_steps = set(sorted(the_big.query("tag == 'time/fps'").step.unique())[::skip_step])  # noqa: F841
    subbig = the_big.query("step in @good_steps or tag in @eval_tags")
    return subbig
