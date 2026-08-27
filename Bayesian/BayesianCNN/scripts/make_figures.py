from lib.plot import plot_param_count_vs_test_acc

if __name__ == "__main__":
    _fashion_acc_experiments = [
        "baseline_25f_25f",
        "baseline_50f_50f",
        "baseline_75f_75f",
        "baseline_100f_100f",
        "plasticity_300f_300f_0",
        "plasticity_300f_300f_1e-07",
        "plasticity_300f_300f_5e-07",
        "plasticity_300f_300f_1e-06",
        "plasticity_300f_300f_5e-06",
        "plasticity_300f_300f_0_prune_only",
        "plasticity_300f_300f_1e-07_prune_only",
        "plasticity_300f_300f_5e-07_prune_only",
        "plasticity_300f_300f_1e-06_prune_only",
        "plasticity_300f_300f_5e-06_prune_only",
    ]

    _fashion_pareto_kinds = ("baseline", "plasticity")

    plot_param_count_vs_test_acc(
        save_path="results_fashionmnist_cnn",
        experiments=_fashion_acc_experiments,
        num_runs=5,
        aggregate_runs=True,
        show_pareto_frontier=True,
        pareto_scope="per_junctures_mode",
        pareto_frontier_kinds=_fashion_pareto_kinds,
        x_col="Parameters",
        xlabel="Parameter Count",
        save_path_out="results_fashionmnist_cnn_acc.png",
        title="Test Acc vs Parameter Count",
        y_col="Test Acc",
        ylabel="Test Acc",
        show=False,
        legend_mode="pareto",
        pareto_linewidth=2,
    )

    plot_param_count_vs_test_acc(
        save_path="results_fashionmnist_cnn",
        experiments=_fashion_acc_experiments,
        num_runs=5,
        aggregate_runs=True,
        show_pareto_frontier=True,
        pareto_scope="per_junctures_mode",
        pareto_frontier_kinds=_fashion_pareto_kinds,
        x_col="Parameters",
        xlabel="Parameter Count",
        save_path_out="results_fashionmnist_cnn_brier.png",
        title="Test Brier vs Parameter Count",
        y_col="Test Brier",
        ylabel="Test Brier",
        show=False,
        legend_mode="pareto",
        pareto_linewidth=2,
    )
