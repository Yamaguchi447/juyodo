import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import optunahub
from optuna.distributions import FloatDistribution

from optuna.samplers.seqUD2 import SeqUD # 重要度を考慮しない元のSeqUDを指定
from optuna.samplers.seqUD5 import SeqUD2 # 比較したい重要度を考慮したSeqUDのファイル名をfromに指定（クラス名は元との分別のため全ファイルともSeqUD2）

from optuna.samplers.ud2 import UniformDesignSampler

from bbob.run_bbob import eval

def sort_data(data, descending=True):
    new_data = []
    for i, val in enumerate(data):
        if i == 0:
            new_data.append(val)
        else:
            if descending:
                if new_data[i - 1] > val and str(val) != "nan":
                    new_data.append(val)
                else:
                    new_data.append(new_data[i - 1])
            else:
                if new_data[i - 1] < val:
                    new_data.append(val)
                else:
                    new_data.append(new_data[i - 1])
    return new_data

class OptunaObjective(object):
    def __init__(self, dim, func_id, seed, X_opt=False, F_opt=False) -> None:
        self.dim = dim
        self.func_id = func_id
        self.seed = seed
        self.X_opt = X_opt
        self.F_opt = F_opt

    def __call__(self, trial):
        X = np.array([trial.suggest_float(f"x{i}", -5, 5) for i in range(self.dim)])
        objective_value = eval(X, self.func_id, self.seed, self.X_opt, self.F_opt)
        return objective_value

def run_experiment():
    output_dir_name = "/Users/yotaroy./Desktop/id07" #保存先のディレクトリを指定

    dim = 10 # 次元数を指定
    func_id = 7 # 関数のidを指定
    X_opt = True
    F_opt = False
    n_trials = 200
    SEED = 1234
    RUN_INSTANCE = 15
    
    
    search_space = {f"x{i}": FloatDistribution(-5, 5) for i in range(dim)}

    param_space = {
        f"x{i}": {"Type": "continuous", "Range": [-5, 5]} for i in range(dim)
    }

    for sampler_name in ["cmaes", "sequd", "sequd2"]:
        np.random.seed(SEED)
        random_seeds = [np.random.randint(1, 1e5) for _ in range(RUN_INSTANCE)]
        output_dir_path = f"{output_dir_name}/sampler_name_{sampler_name}/"
        os.makedirs(output_dir_path, exist_ok=True)
# 個体数を変更する場合は，以下のpopsize, n_runs_per_stageをそれぞれ変更（下記は50の状態）
        for r_seed in random_seeds:
            if sampler_name == "random":
                sampler = optuna.samplers.RandomSampler(seed=SEED)
            elif sampler_name == "cmaes":
                sampler = optuna.samplers.CmaEsSampler(popsize=50, seed=SEED, sigma0=0.5)
            elif sampler_name == "uniform":
                sampler = UniformDesignSampler(
                    search_space=search_space, discretization_level=200
                )
            elif sampler_name == "sequd":
                sampler = SeqUD(
                    param_space, n_runs_per_stage=50, random_state=SEED
                )
            elif sampler_name == "sequd2":
                sampler = SeqUD2(
                    param_space, n_runs_per_stage=50, random_state=SEED
                )
            optuna_objective = OptunaObjective(dim, func_id, r_seed, X_opt, F_opt)

            if sampler_name == "sequd":
                study = optuna.create_study(sampler=sampler, direction="minimize")
                study.optimize(optuna_objective, n_trials=200)
                study_df = study.trials_dataframe()
            elif sampler_name == "sequd2":
                study = optuna.create_study(sampler=sampler, direction="minimize")
                study.optimize(optuna_objective, n_trials=200)
                study_df = study.trials_dataframe()
            else:
                study = optuna.create_study(sampler=sampler, direction="minimize")
                study.optimize(optuna_objective, n_trials=n_trials)
                study_df = study.trials_dataframe()

            study_df.to_csv(output_dir_path + f"r_seed_{r_seed}.csv", index=False)



def plot_result():
    input_dir_path = "/Users/yotaroy./Desktop/id07/" # 保存先のディレクトリを指定
    sampler_names = ["cmaes", "sequd", "sequd2"]
    colors = ["blue", "red", "green"]

    fig, ax = plt.subplots(figsize=(16, 9))
    for sampler_name, COLOR in zip(sampler_names, colors):
        file_path_list = sorted(glob.glob(input_dir_path + f"sampler_name_{sampler_name}/*csv"))

        plot_data = pd.DataFrame()
        for results_index, path in enumerate(file_path_list):
            results = pd.read_csv(path)
            plot_data[results_index] = np.array(sort_data(results["value"]))

        MEAN = plot_data.mean(axis=1)
        SEM = plot_data.sem(axis=1)
        ax.plot(
            np.arange(1, len(plot_data) + 1),
            MEAN,
            color=COLOR,
            linewidth=4,
            alpha=0.8,
            label=f" {sampler_name.capitalize()}",
        )
        ax.fill_between(
            np.arange(1, len(plot_data) + 1),
            MEAN + SEM,
            MEAN - SEM,
            color=COLOR,
            alpha=0.4,
            linewidth=0,
        )

    ax.tick_params(labelsize=25)
    ax.set_xlim(1, 200)
    ax.set_xlabel("Function evaluations", fontsize=25)
    ax.set_ylim(0, 1000) # 縦軸はid7の場合1000，id2&10の場合は3000000程度
    ax.set_ylabel("Objective", fontsize=25)
    ax.grid()
    ax.legend(
        bbox_to_anchor=(1, 1),
        loc="upper right",
        borderaxespad=0.2,
        fontsize=25,
    )
    plt.tight_layout()
    plt.savefig(input_dir_path + "results.png")
    plt.close()

if __name__ == "__main__":
    run_experiment()
    plot_result()