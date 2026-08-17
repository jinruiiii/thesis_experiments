from experiments.run_plasticity import _format_lambda_dir
from experiments.run_shift import main
from lib.plot import plot_param_count_vs_test_acc
from lib.seed import SEED, set_seed

if __name__ == "__main__":
    # Example: same-start static baseline + one plasticity lambda.
    phase1_epochs = 30
    phase2_epochs = 30

    hidden_sizes = [500, 500]
    for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
        for i in range(1, 6):
            set_seed(SEED+i)
            main(
                f"results_prior_shift_FashionMnist_FNN_new/run_{i}",
                hidden_sizes,
                lambda_penalty=lambda_penalty,
                run_mode="plasticity",
                phase2_only_junctures=False,
                phase2_regrow_to_init=False,
                phase2_vcl_prior=True,
                phase2_juncture_warmup_epochs=0,
                phase1_epochs=phase1_epochs,
                phase2_epochs=phase2_epochs,
            )
    for lambda_penalty in [0, 1e-07, 5e-07, 1e-06, 5e-06]:
        for i in range(1, 6):
            set_seed(SEED + i)
            lam_tag = _format_lambda_dir(lambda_penalty)
            main(
                f"results_prior_shift_FashionMnist_FNN_new/run_{i}",
                hidden_sizes,
                run_mode="static_replay",
                resume_from_plasticity_dir=(
                    f"results_prior_shift_FashionMnist_FNN_new/run_{i}/"
                    f"plasticity_500_{lam_tag}_vcl"
                ),
                phase2_vcl_prior=True,
                phase1_epochs=phase1_epochs,
                phase2_epochs=phase2_epochs,
            )




    # hidden_sizes = [500, 500]
    # for lambda_penalty in [0, 5e-08, 5e-07, 1e-06, 5e-06]:
    #     for i in range(1, 6):
    #         set_seed(SEED+i)
    #         main(
    #             f"results_prior_shift_FashionMnist_FNN1/run_{i}",
    #             hidden_sizes,
    #             lambda_penalty=lambda_penalty,
    #             run_mode="plasticity",
    #             phase2_regrow_to_init=True,
    #             phase2_vcl_prior=True,
    #             phase2_juncture_warmup_epochs=0,
    #             phase1_epochs=phase1_epochs,
    #             phase2_epochs=phase2_epochs,
    #         )

    # Same-end static_replay control after plasticity_*_vcl runs (mirrors main2 naming):
    # writes static_replay_500_<lambda>_vcl beside each source plasticity folder.
    # hidden_sizes = [500, 500]  # API-only; widths come from plasticity summary
    # for lambda_penalty in [0, 5e-08, 1e-07, 5e-07, 1e-06, 5e-06]:
    #     for i in range(1, 6):
    #         set_seed(SEED + i)
    #         lam_tag = _format_lambda_dir(lambda_penalty)
    #         main(
    #             f"results_prior_shift_FashionMnist_FNN_reverse_new/run_{i}",
    #             hidden_sizes,
    #             run_mode="static_replay",
    #             resume_from_plasticity_dir=(
    #                 f"results_prior_shift_FashionMnist_FNN_reverse_new/run_{i}/"
    #                 f"plasticity_500_{lam_tag}_vcl"
    #             ),
    #             phase2_vcl_prior=True,
    #             phase1_epochs=phase1_epochs,
    #             phase2_epochs=phase2_epochs,
    #         )

    # plot_param_count_vs_test_acc(
    #     save_path="results_prior_shift_FashionMnist_FNN",
    #     experiments=[
    #         "baseline_20_vcl","baseline_50_vcl", "baseline_100_vcl", "baseline_150_vcl", "baseline_200_vcl",
    #         "plasticity_500_0_vcl","plasticity_500_5e-08_vcl","plasticity_500_1e-07_vcl", "plasticity_500_5e-07_vcl", "plasticity_500_1e-06_vcl", "plasticity_500_5e-06_vcl",
    #         # "static_replay_500_0_vcl","static_replay_500_5e-08_vcl", "static_replay_500_1e-07_vcl","static_replay_500_5e-07_vcl", "static_replay_500_1e-06_vcl", "static_replay_500_5e-06_vcl",
    #         # "plasticity_500_0_regrow_vcl","plasticity_500_5e-08_regrow_vcl", "plasticity_500_5e-07_regrow_vcl", "plasticity_500_1e-06_regrow_vcl", "plasticity_500_5e-06_regrow_vcl",
    #         # "plasticity_500_0_p2junct_vcl","plasticity_500_5e-08_p2junct_vcl", "plasticity_500_5e-07_p2junct_vcl", "plasticity_500_1e-06_p2junct_vcl", "plasticity_500_5e-06_p2junct_vcl",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     show_pareto_frontier=True,
    #     pareto_scope="per_kind",
    #     pareto_frontier_kinds=("baseline", "plasticity", "static_replay"),
    #     y_col="Test Acc",  # or "Test Group B Acc" / "Test Group A Acc"
    #     title="Test Acc (Balanced) vs Parameter Count",
    #     save_path_out="results_prior_shift_FashionMnist_FNN/balanced_test_acc_vcl.png",
    # )
    # plot_param_count_vs_test_acc(
    #     save_path="results_prior_shift_FashionMnist_FNN",
    #     experiments=[
    #         "baseline_20_vcl","baseline_50_vcl", "baseline_100_vcl", "baseline_150_vcl", "baseline_200_vcl",
    #         "plasticity_500_0_vcl","plasticity_500_5e-08_vcl","plasticity_500_1e-07_vcl", "plasticity_500_5e-07_vcl", "plasticity_500_1e-06_vcl", "plasticity_500_5e-06_vcl",
    #         # "static_replay_500_0_vcl","static_replay_500_5e-08_vcl", "static_replay_500_1e-07_vcl","static_replay_500_5e-07_vcl", "static_replay_500_1e-06_vcl", "static_replay_500_5e-06_vcl",
    #         # "plasticity_500_0_regrow_vcl","plasticity_500_5e-08_regrow_vcl", "plasticity_500_5e-07_regrow_vcl", "plasticity_500_1e-06_regrow_vcl", "plasticity_500_5e-06_regrow_vcl",
    #         # "plasticity_500_0_p2junct_vcl","plasticity_500_5e-08_p2junct_vcl", "plasticity_500_5e-07_p2junct_vcl", "plasticity_500_1e-06_p2junct_vcl", "plasticity_500_5e-06_p2junct_vcl",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     show_pareto_frontier=True,
    #     pareto_scope="per_kind",
    #     pareto_frontier_kinds=("baseline", "plasticity", "static_replay"),
    #     y_col="Test Brier",  # or "Test Group B Acc" / "Test Group A Acc"
    #     title="Test Brier (Balanced)vs Parameter Count",
    #     save_path_out="results_prior_shift_FashionMnist_FNN/balanced_test_brier_vcl.png",
    # )
    # plot_param_count_vs_test_acc(
    #     save_path="results_prior_shift_FashionMnist_FNN",
    #     experiments=[
    #         "baseline_20_vcl","baseline_50_vcl", "baseline_100_vcl", "baseline_150_vcl", "baseline_200_vcl",
    #         "plasticity_500_0_vcl","plasticity_500_5e-08_vcl", "plasticity_500_5e-07_vcl", "plasticity_500_1e-06_vcl", "plasticity_500_5e-06_vcl",
    #         # "static_replay_500_0_vcl","static_replay_500_5e-08_vcl", "static_replay_500_5e-07_vcl", "static_replay_500_1e-06_vcl", "static_replay_500_5e-06_vcl",
    #         # "plasticity_500_0_regrow_vcl","plasticity_500_5e-08_regrow_vcl", "plasticity_500_5e-07_regrow_vcl", "plasticity_500_1e-06_regrow_vcl", "plasticity_500_5e-06_regrow_vcl",
    #         # "plasticity_500_0_p2junct_vcl","plasticity_500_5e-08_p2junct_vcl", "plasticity_500_5e-07_p2junct_vcl", "plasticity_500_1e-06_p2junct_vcl", "plasticity_500_5e-06_p2junct_vcl",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     show_pareto_frontier=True,
    #     pareto_scope="per_kind",
    #     pareto_frontier_kinds=("baseline", "plasticity", "static_replay"),
    #     y_col="Phase2 Matched Test Acc",  # or "Test Group B Acc" / "Test Group A Acc"
    #     title="Test Acc (Phase2 Prior)vs Parameter Count",
    #     save_path_out="results_prior_shift_FashionMnist_FNN/phase2_matched_test_acc_vcl.png",
    # )
    # plot_param_count_vs_test_acc(
    #     save_path="results_prior_shift_FashionMnist_FNN",
    #     experiments=[
    #         "baseline_20_vcl","baseline_50_vcl", "baseline_100_vcl", "baseline_150_vcl", "baseline_200_vcl",
    #         "plasticity_500_0_vcl","plasticity_500_5e-08_vcl", "plasticity_500_5e-07_vcl", "plasticity_500_1e-06_vcl", "plasticity_500_5e-06_vcl",
    #         # "static_replay_500_0_vcl","static_replay_500_5e-08_vcl", "static_replay_500_5e-07_vcl", "static_replay_500_1e-06_vcl", "static_replay_500_5e-06_vcl",
    #         # "plasticity_500_0_regrow_vcl","plasticity_500_5e-08_regrow_vcl", "plasticity_500_5e-07_regrow_vcl", "plasticity_500_1e-06_regrow_vcl", "plasticity_500_5e-06_regrow_vcl",
    #         # "plasticity_500_0_p2junct_vcl","plasticity_500_5e-08_p2junct_vcl", "plasticity_500_5e-07_p2junct_vcl", "plasticity_500_1e-06_p2junct_vcl", "plasticity_500_5e-06_p2junct_vcl",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     show_pareto_frontier=True,
    #     pareto_scope="per_kind",
    #     pareto_frontier_kinds=("baseline", "plasticity", "static_replay"),
    #     y_col="Phase2 Matched Test Brier",  # or "Test Group B Acc" / "Test Group A Acc"
    #     title="Test Brier (Phase 2 Prior) vs Parameter Count",
    #     save_path_out="results_prior_shift_FashionMnist_FNN/phase2_matched_test_brier_vcl.png",
    # )
    # plot_param_count_vs_test_acc(
    #     save_path="results_prior_shift_FashionMnist_FNN_reverse",
    #     experiments=[
    #         "baseline_20_vcl","baseline_50_vcl", "baseline_100_vcl", "baseline_150_vcl", "baseline_200_vcl",
    #         "plasticity_500_0_vcl","plasticity_500_5e-08_vcl", "plasticity_500_5e-07_vcl", "plasticity_500_1e-06_vcl", "plasticity_500_5e-06_vcl",
    #         "static_replay_500_0_vcl","static_replay_500_5e-08_vcl", "static_replay_500_5e-07_vcl", "static_replay_500_1e-06_vcl", "static_replay_500_5e-06_vcl",
    #         # "plasticity_500_0_regrow_vcl","plasticity_500_5e-08_regrow_vcl", "plasticity_500_5e-07_regrow_vcl", "plasticity_500_1e-06_regrow_vcl", "plasticity_500_5e-06_regrow_vcl",
    #         # "plasticity_500_0_p2junct_vcl","plasticity_500_5e-08_p2junct_vcl", "plasticity_500_5e-07_p2junct_vcl", "plasticity_500_1e-06_p2junct_vcl", "plasticity_500_5e-06_p2junct_vcl",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     show_pareto_frontier=True,
    #     pareto_scope="per_kind",
    #     pareto_frontier_kinds=("baseline", "plasticity", "static_replay"),
    #     y_col="Phase1 Matched Forget Acc",  # or "Test Group B Acc" / "Test Group A Acc"
    #     title="Test Acc vs Parameter Count",
    #     save_path_out="results_prior_shift_FashionMnist_FNN_reverse/phase1_matched_forget_test_acc_vcl.png",
    # )

