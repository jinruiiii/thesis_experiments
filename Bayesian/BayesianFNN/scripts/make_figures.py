from experiments.run_plasticity import _format_lambda_dir, main
from lib.plot import get_statistics, plot_param_count, plot_param_count_vs_test_acc,plot_structural_action_proportions, plot_structural_decision_heatmap
from lib.seed import SEED, set_seed

if __name__ == "__main__":
    
    _fashion_acc_experiments = [
        "baseline_20", "baseline_50", "baseline_100", "baseline_150", "baseline_200",
        "three_phase_baseline_20","three_phase_baseline_50","three_phase_baseline_100","three_phase_baseline_150","three_phase_baseline_200",
        "plasticity_500_5e-06", "plasticity_500_1e-06", "plasticity_500_5e-07",
        "plasticity_500_1e-07", "plasticity_500_0",
        "plasticity_500_5e-06_prune_only", "plasticity_500_1e-06_prune_only",
        "plasticity_500_5e-07_prune_only", "plasticity_500_1e-07_prune_only",
        "plasticity_500_0_prune_only",
        "static_replay_500_5e-06", "static_replay_500_1e-06",
        "static_replay_500_5e-07", "static_replay_500_1e-07", "static_replay_500_0",
        "nest_300_100_p0.1_refacc89_flooracc87",
        "nest_300_100_p0.1_refacc89_flooracc87.5",
        "nest_300_100_p0.1_refacc89_flooracc88",
        "nest_300_100_p0.1_refacc89_flooracc88.5",
        "nest_300_100_p0.1_refacc89_flooracc89",
    ]

    _fashion_acc_experiments_shift = [
        "baseline_20_vcl", "baseline_50_vcl", "baseline_100_vcl", "baseline_150_vcl", "baseline_200_vcl",
        "plasticity_500_5e-06_vcl", "plasticity_500_1e-06_vcl", "plasticity_500_5e-07_vcl",
        "plasticity_500_1e-07_vcl", "plasticity_500_0_vcl",
        "static_replay_500_5e-06_vcl", "static_replay_500_1e-06_vcl",
        "static_replay_500_5e-07_vcl", "static_replay_500_1e-07_vcl", "static_replay_500_0_vcl",
    ]
    _fashion_pareto_kinds = (
        "baseline", "three_phase", "plasticity", "plasticity_three_phase",
        "static_replay", "nest", "nest_dense", "dropnet",
    )

    plot_param_count_vs_test_acc(
        save_path="results_fashionmnist",
        experiments=_fashion_acc_experiments,
        num_runs=5,
        aggregate_runs=True,
        show_pareto_frontier=True,
        pareto_scope="per_junctures_mode",
        pareto_frontier_kinds=_fashion_pareto_kinds,
        x_col="Parameters",
        xlabel="Parameter Count",
        save_path_out="results_fashionmnist_acc.png",
        title="Test Acc vs Parameter Count",
        y_col="Test Acc",
        ylabel="Test Acc",
        show=False,
        legend_mode="pareto",
        axis_label_fontsize=20,
        tick_label_fontsize=20,
        title_fontsize=20,
        legend_fontsize=20,
        pareto_linewidth=5,
        nest_x_modes=("sparse", "dense"),
        x_tick_interval=100_000,
    )

    plot_param_count_vs_test_acc(
        save_path="results_fashionmnist",
        experiments=_fashion_acc_experiments,
        num_runs=5,
        aggregate_runs=True,
        show_pareto_frontier=True,
        pareto_scope="per_junctures_mode",
        pareto_frontier_kinds=_fashion_pareto_kinds,
        x_col="Parameters",
        xlabel="Parameter Count",
        save_path_out="results_fashionmnist_brier.png",
        title="Test Brier vs Parameter Count",
        y_col="Test Brier",
        ylabel="Test Brier",
        show=False,
        legend_mode="pareto",
        axis_label_fontsize=20,
        tick_label_fontsize=20,
        title_fontsize=20,
        legend_fontsize=20,
        pareto_linewidth=5,
        nest_x_modes=("sparse", "dense"),
        x_tick_interval=100_000,
    )


    plot_param_count(
        save_path="results_fashionmnist",
        experiments=[
            "plasticity_500_0","plasticity_500_1e-07","plasticity_500_5e-07","plasticity_500_1e-06","plasticity_500_5e-06"
        ],
        show_individual=False,
        show_checkpoint=True,
        save_path_out="results_fashionmnist_param_count_vs_epoch.png",
        axis_label_fontsize=20,
        tick_label_fontsize=20,
        title_fontsize=20,
        legend_fontsize=20,
        show=True,
        yoffset_fontsize=20,
    )



    plot_structural_decision_heatmap(
        save_path="results_prior_shift_fashionmnist",
        experiments=[
            "plasticity_500_0",
            "plasticity_500_1e-07",
            "plasticity_500_5e-07",
            "plasticity_500_1e-06",
            "plasticity_500_5e-06",
        ],
        save_path_out="decision_heatmap.png",
        axis_label_fontsize=20,
        tick_label_fontsize=16,
        title_fontsize=20,
        legend_fontsize=16,
        ylabels="full",
        show=False,
    )