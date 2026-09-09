from experiments.run_plasticity import _format_lambda_dir, main
from lib.plot import plot_param_count, plot_param_count_vs_test_acc, plot_structural_decision_heatmap, plot_gamma_pareto_comparison
from lib.seed import SEED, set_seed

if __name__ == "__main__":
    
    _experiments = [
        "plasticity_500_0", "plasticity_500_1e-07", "plasticity_500_5e-07",
        "plasticity_500_1e-06", "plasticity_500_5e-06",
    ]

    _cifar_static_experiments = [
        "baseline_30","baseline_50","baseline_100", "baseline_150", "baseline_200",
        "plasticity_500_5e-06", "plasticity_500_1e-06", "plasticity_500_5e-07",
        "plasticity_500_1e-07", "plasticity_500_0",
        "plasticity_500_5e-06_prune_only", "plasticity_500_1e-06_prune_only",
        "plasticity_500_5e-07_prune_only", "plasticity_500_1e-07_prune_only",
        "plasticity_500_0_prune_only",
        "static_replay_500_5e-06", "static_replay_500_1e-06",
        "static_replay_500_5e-07", "static_replay_500_1e-07", "static_replay_500_0",
    ]

    _cifar_dynamic_experiments = [
        "three_phase_baseline_30","three_phase_baseline_50","three_phase_baseline_100","three_phase_baseline_150","three_phase_baseline_200",
        "plasticity_500_5e-06", "plasticity_500_1e-06", "plasticity_500_5e-07",
        "plasticity_500_1e-07", "plasticity_500_0",
        "nest_300_100_p0.1_refacc44_flooracc42",
        "nest_300_100_p0.1_refacc44_flooracc42.5",
        "nest_300_100_p0.1_refacc44_flooracc43",
        "nest_300_100_p0.1_refacc44_flooracc43.5",
        "nest_300_100_p0.1_refacc44_flooracc44",
    ]

    _cifar_all_experiments = [
        "baseline_30","baseline_50","baseline_100", "baseline_150", "baseline_200",
        "plasticity_500_5e-06", "plasticity_500_1e-06", "plasticity_500_5e-07",
        "plasticity_500_1e-07", "plasticity_500_0",
        "plasticity_500_5e-06_prune_only", "plasticity_500_1e-06_prune_only",
        "plasticity_500_5e-07_prune_only", "plasticity_500_1e-07_prune_only",
        "plasticity_500_0_prune_only",
        "static_replay_500_5e-06", "static_replay_500_1e-06",
        "static_replay_500_5e-07", "static_replay_500_1e-07", "static_replay_500_0",
        "three_phase_baseline_30","three_phase_baseline_50","three_phase_baseline_100","three_phase_baseline_150","three_phase_baseline_200",
        "nest_300_100_p0.1_refacc44_flooracc42",
        "nest_300_100_p0.1_refacc44_flooracc42.5",
        "nest_300_100_p0.1_refacc44_flooracc43",
        "nest_300_100_p0.1_refacc44_flooracc43.5",
        "nest_300_100_p0.1_refacc44_flooracc44",

    ]

    _cifar_pareto_kinds = (
        "baseline", "three_phase", "plasticity", "plasticity_three_phase",
        "static_replay", "nest", "nest_dense", "dropnet",
    )

    _pareto_plot_kwargs = dict(
        save_path="results_cifar10",
        num_runs=5,
        aggregate_runs=True,
        show_pareto_frontier=True,
        pareto_scope="per_junctures_mode",
        pareto_frontier_kinds=_cifar_pareto_kinds,
        x_col="Parameters",
        xlabel="Parameter Count",
        show=False,
        legend_mode="pareto",
        pareto_linewidth=5,
        axis_label_fontsize=20,
        tick_label_fontsize=20,
        title_fontsize=20,
        legend_fontsize=20,
        nest_x_modes=("sparse"),
        x_tick_interval=100_000,
        show_points=False,
        marker_size=120,
    )

    plot_param_count_vs_test_acc(
        experiments=_cifar_static_experiments,
        save_path_out="cifar10_plots/static/static_accuracy.pdf",
        title="Test Acc vs Parameter Count",
        y_col="Test Acc",
        ylabel="Test Acc",
        **_pareto_plot_kwargs,
    )

    plot_param_count_vs_test_acc(
        experiments=_cifar_static_experiments,
        save_path_out="cifar10_plots/static/static_brier.pdf",
        title="Test Brier vs Parameter Count",
        y_col="Test Brier",
        ylabel="Test Brier",
        **_pareto_plot_kwargs,
    )

    plot_param_count_vs_test_acc(
        experiments=_cifar_dynamic_experiments,
        save_path_out="cifar10_plots/dynamic/dynamic_accuracy.pdf",
        title="Test Acc vs Parameter Count",
        y_col="Test Acc",
        ylabel="Test Acc",
        **_pareto_plot_kwargs,
    )

    plot_param_count_vs_test_acc(
        experiments=_cifar_dynamic_experiments,
        save_path_out="cifar10_plots/dynamic/dynamic_brier.pdf",
        title="Test Brier vs Parameter Count",
        y_col="Test Brier",
        ylabel="Test Brier",
        **_pareto_plot_kwargs,
    )

    plot_param_count_vs_test_acc(
        experiments=_cifar_all_experiments,
        save_path_out="cifar10_plots/all/all_accuracy.pdf",
        title="Test Acc vs Parameter Count",
        y_col="Test Acc",
        ylabel="Test Acc",
        **_pareto_plot_kwargs,
    )

    plot_param_count_vs_test_acc(
        experiments=_cifar_all_experiments,
        save_path_out="cifar10_plots/all/all_brier.pdf",
        title="Test Brier vs Parameter Count",
        y_col="Test Brier",
        ylabel="Test Brier",
        **_pareto_plot_kwargs,
    )

    plot_param_count(
        save_path="results_cifar10",
        experiments=_experiments,
        show_individual=False,
        show_checkpoint=True,
        checkpoint_tail_epochs=5,
        save_path_out="cifar10_plots/structure/param_count_vs_epoch_diff_lambda.pdf",
        axis_label_fontsize=20,
        tick_label_fontsize=20,
        title_fontsize=20,
        legend_fontsize=20,
        show=True,
        yoffset_fontsize=20,
    )


    plot_structural_decision_heatmap(
        save_path="results_cifar10",
        experiments=_experiments,
        save_path_out="cifar10_plots/structure/heatmap.pdf",
        axis_label_fontsize=20,
        tick_label_fontsize=16,
        title_fontsize=20,
        legend_fontsize=16,
        ylabels="full",
        show=False,
    )

