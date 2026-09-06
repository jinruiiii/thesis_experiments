from lib.plot import (
    plot_param_count,
    plot_param_count_vs_test_acc,
    plot_structural_decision_heatmap,
)

if __name__ == "__main__":

    _experiments = [
        "plasticity_300f_300f_1e-07",
        "plasticity_300f_300f_5e-07",
        "plasticity_300f_300f_1e-06",
        "plasticity_300f_300f_2.5e-06",
        "plasticity_300f_300f_5e-06",
    ]

    _cifar_static_experiments = [
        "baseline_30f_30f",
        "baseline_40f_40f",
        "baseline_60f_60f",
        "baseline_80f_80f",
        "baseline_100f_100f",
        "plasticity_300f_300f_1e-07",
        "plasticity_300f_300f_5e-07",
        "plasticity_300f_300f_1e-06",
        "plasticity_300f_300f_2.5e-06",
        "plasticity_300f_300f_5e-06",
        "plasticity_300f_300f_1e-07_prune_only",
        "plasticity_300f_300f_5e-07_prune_only",
        "plasticity_300f_300f_1e-06_prune_only",
        "plasticity_300f_300f_2.5e-06_prune_only",
        "plasticity_300f_300f_5e-06_prune_only",
    ]

    _cifar_dynamic_experiments = [
        "plasticity_300f_300f_1e-07",
        "plasticity_300f_300f_5e-07",
        "plasticity_300f_300f_1e-06",
        "plasticity_300f_300f_2.5e-06",
        "plasticity_300f_300f_5e-06",
        "three_phase_baseline_30f_30f",
        "three_phase_baseline_40f_40f",
        "three_phase_baseline_60f_60f",
        "three_phase_baseline_80f_80f",
        "three_phase_baseline_100f_100f",
        "static_replay_300f_300f_1e-07",
        "static_replay_300f_300f_5e-07",
        "static_replay_300f_300f_1e-06",
        "static_replay_300f_300f_2.5e-06",
        "static_replay_300f_300f_5e-06",
        "nest_300f_100f_p0.1_refacc71.5_flooracc70.5",
        "nest_300f_100f_p0.1_refacc71.5_flooracc70.75",
        "nest_300f_100f_p0.1_refacc71.5_flooracc71",
        "nest_300f_100f_p0.1_refacc71.5_flooracc71.25",
        "nest_300f_100f_p0.1_refacc71.5_flooracc71.5",
    ]

    _cifar_all_experiments = [
        "baseline_30f_30f",
        "baseline_40f_40f",
        "baseline_60f_60f",
        "baseline_80f_80f",
        "baseline_100f_100f",
        "plasticity_300f_300f_1e-07",
        "plasticity_300f_300f_5e-07",
        "plasticity_300f_300f_1e-06",
        "plasticity_300f_300f_2.5e-06",
        "plasticity_300f_300f_5e-06",
        "plasticity_300f_300f_1e-07_prune_only",
        "plasticity_300f_300f_5e-07_prune_only",
        "plasticity_300f_300f_1e-06_prune_only",
        "plasticity_300f_300f_2.5e-06_prune_only",
        "plasticity_300f_300f_5e-06_prune_only",
        "three_phase_baseline_30f_30f",
        "three_phase_baseline_40f_40f",
        "three_phase_baseline_60f_60f",
        "three_phase_baseline_80f_80f",
        "three_phase_baseline_100f_100f",
        "static_replay_300f_300f_1e-07",
        "static_replay_300f_300f_5e-07",
        "static_replay_300f_300f_1e-06",
        "static_replay_300f_300f_2.5e-06",
        "static_replay_300f_300f_5e-06",
        "nest_300f_100f_p0.1_refacc71.5_flooracc70.5",
        "nest_300f_100f_p0.1_refacc71.5_flooracc70.75",
        "nest_300f_100f_p0.1_refacc71.5_flooracc71",
        "nest_300f_100f_p0.1_refacc71.5_flooracc71.25",
        "nest_300f_100f_p0.1_refacc71.5_flooracc71.5",
    ]

    _cifar_shift_experiments = [
        "baseline_30f_30f_vcl",
        "baseline_40f_40f_vcl",
        "baseline_60f_60f_vcl",
        "baseline_80f_80f_vcl",
        "baseline_100f_100f_vcl",
        "plasticity_300f_300f_1e-07_vcl",
        "plasticity_300f_300f_5e-07_vcl",
        "plasticity_300f_300f_1e-06_vcl",
        "plasticity_300f_300f_2.5e-06_vcl",
        "plasticity_300f_300f_5e-06_vcl",
    ]

    _cifar_pareto_kinds = (
        "baseline",
        "three_phase",
        "plasticity",
        "static_replay",
        "nest",
        "nest_dense",
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
        show=False,
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

    _cifar_random_growth_compare = [
        "plasticity_300f_300f_1e-07",
        "plasticity_300f_300f_5e-07",
        "plasticity_300f_300f_1e-06",
        "plasticity_300f_300f_2.5e-06",
        "plasticity_300f_300f_5e-06",
        "plasticity_300f_300f_1e-07_random_grow",
        "plasticity_300f_300f_5e-07_random_grow",
        "plasticity_300f_300f_1e-06_random_grow",
        "plasticity_300f_300f_2.5e-06_random_grow",
        "plasticity_300f_300f_5e-06_random_grow",
    ]

    plot_param_count_vs_test_acc(
        experiments=_cifar_random_growth_compare,
        save_path_out="cifar10_plots/random_growth/acc.pdf",
        title="Test Acc vs Parameter Count",
        y_col="Test Acc",
        ylabel="Test Acc",
        **{
            **_pareto_plot_kwargs,
            "pareto_frontier_kinds": ("plasticity", "plasticity_random"),
        },
    )

    plot_param_count_vs_test_acc(
        experiments=_cifar_random_growth_compare,
        save_path_out="cifar10_plots/random_growth/brier.pdf",
        title="Test Brier vs Parameter Count",
        y_col="Test Brier",
        ylabel="Test Brier",
        **{
            **_pareto_plot_kwargs,
            "pareto_frontier_kinds": ("plasticity", "plasticity_random"),
        },
    )

