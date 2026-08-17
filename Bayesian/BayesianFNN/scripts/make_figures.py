"""Rebuild thesis figures from saved experiment_summary.csv files."""

from lib.plot import plot_param_count_vs_test_acc


def plot_fashion_mnist_pareto():
    experiments = [
        "baseline_20",
        "baseline_50",
        "baseline_100",
        "baseline_150",
        "baseline_200",
        "plasticity_500_5e-06",
        "plasticity_500_1e-06",
        "plasticity_500_5e-07",
        "plasticity_500_1e-07",
        "plasticity_500_0",
        "static_replay_500_5e-06",
        "static_replay_500_1e-06",
        "static_replay_500_5e-07",
        "static_replay_500_1e-07",
        "static_replay_500_0",
    ]
    common = dict(
        save_path="results_FashionMnist_FNN",
        experiments=experiments,
        num_runs=5,
        aggregate_runs=True,
        show_pareto_frontier=True,
        pareto_scope="per_junctures_mode",
        pareto_frontier_kinds=(
            "baseline",
            "three_phase",
            "plasticity",
            "plasticity_three_phase",
            "static_replay",
            "nest",
            "dropnet",
        ),
    )
    plot_param_count_vs_test_acc(
        **common,
        save_path_out="results_FashionMnist_FNN_acc.png",
        title="Test Acc vs Parameter Count",
        y_col="Test Acc",
        ylabel="Test Acc",
    )
    plot_param_count_vs_test_acc(
        **common,
        save_path_out="results_FashionMnist_FNN_brier.png",
        title="Test Brier vs Parameter Count",
        y_col="Test Brier",
        ylabel="Test Brier",
    )


def plot_prior_shift_vcl():
    experiments = [
        "baseline_20_vcl",
        "baseline_50_vcl",
        "baseline_100_vcl",
        "baseline_150_vcl",
        "baseline_200_vcl",
        "plasticity_500_0_vcl",
        "plasticity_500_5e-08_vcl",
        "plasticity_500_1e-07_vcl",
        "plasticity_500_5e-07_vcl",
        "plasticity_500_1e-06_vcl",
        "plasticity_500_5e-06_vcl",
    ]
    common = dict(
        save_path="results_prior_shift_FashionMnist_FNN",
        num_runs=5,
        aggregate_runs=True,
        show_pareto_frontier=True,
        pareto_scope="per_kind",
        pareto_frontier_kinds=("baseline", "plasticity", "static_replay"),
    )
    plot_param_count_vs_test_acc(
        **common,
        experiments=experiments,
        y_col="Test Acc",
        title="Test Acc (Balanced) vs Parameter Count",
        save_path_out="results_prior_shift_FashionMnist_FNN/balanced_test_acc_vcl.png",
    )
    plot_param_count_vs_test_acc(
        **common,
        experiments=experiments,
        y_col="Test Brier",
        title="Test Brier (Balanced)vs Parameter Count",
        save_path_out="results_prior_shift_FashionMnist_FNN/balanced_test_brier_vcl.png",
    )
    plot_param_count_vs_test_acc(
        **common,
        experiments=experiments,
        y_col="Phase2 Matched Test Acc",
        title="Test Acc (Phase2 Prior)vs Parameter Count",
        save_path_out="results_prior_shift_FashionMnist_FNN/phase2_matched_test_acc_vcl.png",
    )
    plot_param_count_vs_test_acc(
        **common,
        experiments=experiments,
        y_col="Phase2 Matched Test Brier",
        title="Test Brier (Phase 2 Prior) vs Parameter Count",
        save_path_out="results_prior_shift_FashionMnist_FNN/phase2_matched_test_brier_vcl.png",
    )


if __name__ == "__main__":
    plot_fashion_mnist_pareto()
    plot_prior_shift_vcl()
