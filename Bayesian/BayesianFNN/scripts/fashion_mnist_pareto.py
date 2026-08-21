from experiments.run_plasticity import _format_lambda_dir, main
from lib.plot import get_statistics, plot_param_count, plot_param_count_vs_test_acc,plot_structural_action_proportions, plot_structural_decision_heatmap
from lib.seed import SEED, set_seed

if __name__ == "__main__":
    hidden_sizes = [200, 200]
    for lambda_penalty in [0,5e-07,1e-06,5e-06]:
        for junctures_mode in ["both"]:
            for i in range(1, 6):
                set_seed(SEED+i)
                print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
                main(
                    f"results_fashionmnist_different_initial/run_{i}",
                    hidden_sizes,
                    lambda_penalty,
                    junctures_mode=junctures_mode,
                    dataset="fashion_mnist",
                    run_mode="plasticity",
                )

    hidden_sizes = [100, 100]
    for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
        for junctures_mode in ["both"]:
            for i in range(1, 6):
                set_seed(SEED+i)
                print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
                main(
                    f"results_fashionmnist_different_initial/run_{i}",
                    hidden_sizes,
                    lambda_penalty,
                    junctures_mode=junctures_mode,
                    dataset="fashion_mnist",
                    run_mode="plasticity",
                )

    hidden_sizes = [50, 50]
    for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
        for junctures_mode in ["both"]:
            for i in range(1, 6):
                set_seed(SEED+i)
                print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
                main(
                    f"results_fashionmnist_different_initial/run_{i}",
                    hidden_sizes,
                    lambda_penalty,
                    junctures_mode=junctures_mode,
                    dataset="fashion_mnist",
                    run_mode="plasticity",
                )

    hidden_sizes = [20, 20]
    for lambda_penalty in [0,5e-07,1e-06,5e-06]:
        for junctures_mode in ["both"]:
            for i in range(1, 6):
                set_seed(SEED+i)
                print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
                main(
                    f"results_fashionmnist_different_initial/run_{i}",
                    hidden_sizes,
                    lambda_penalty,
                    junctures_mode=junctures_mode,
                    dataset="fashion_mnist",
                    run_mode="plasticity",
                )

    # hidden_sizes = [500, 500]
    # for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
    #     for junctures_mode in ["prune"]:
    #         for i in range(1, 6):
    #             set_seed(SEED+i)
    #             print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
    #             main(
    #                 f"results_kmnist_new_lr0.001_ws1_ws32/run_{i}",
    #                 hidden_sizes,
    #                 lambda_penalty,
    #                 junctures_mode=junctures_mode,
    #                 dataset="kmnist",
    #                 run_mode="plasticity",
    #             )


    # for hidden_size in [[20,20],[50,50],[100,100],[150,150],[200,200]]:
    #     for lambda_penalty in [0]:
    #         for junctures_mode in ["both"]:
    #             for i in range(1,6):
    #                 set_seed(SEED+i)
    #                 print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
    #                 main(
    #                     f"results_kmnist_new_lr0.001/run_{i}",
    #                     hidden_size,
    #                     lambda_penalty,
    #                     junctures_mode=junctures_mode,
    #                     dataset="kmnist",
    #                     run_mode="baseline",
    #                 )
    # print("All experiments completed.")



    # for hidden_size in [[200,200],[150,150],[100,100],[50,50],[20,20]]:
    #     for lambda_penalty in [0]:
    #         for junctures_mode in ["both"]:
    #             for i in range(1,6):
    #                 set_seed(SEED+i)
    #                 print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
    #                 main(
    #                     f"results_kmnist_new_lr0.001/run_{i}",
    #                     hidden_size,
    #                     lambda_penalty,
    #                     junctures_mode=junctures_mode,
    #                     dataset="kmnist",
    #                     run_mode="three_phase",
    #                     three_phase_growth_layer_idx=1,
    #                     three_phase_growth_gamma=1,
    #                 )
    # print("All experiments completed.")


    # # Static replay: train a fixed FNN using Hidden Sizes from each plasticity run's summary.
    # for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
    #     for i in range(1, 6):
    #         set_seed(SEED+i)
    #         plasticity_dir = (
    #             f"results_kmnist_new/run_{i}/plasticity_500_{_format_lambda_dir(lambda_penalty)}"
    #         )
    #         print(
    #             "Running static replay for run",
    #             i,
    #             f"(source={plasticity_dir})",
    #         )
    #         main(
    #             f"results_kmnist_new/run_{i}",
    #             [500, 500],  # unused for architecture; taken from plasticity summary
    #             lambda_penalty=0,
    #             dataset="kmnist",
    #             run_mode="static_replay",
    #             resume_from_plasticity_dir=plasticity_dir,
    #         )
    # print("All static replay experiments completed.")

    # plot_param_count_vs_test_acc(
    #     save_path="results_kmnist_new",
    #     experiments=[
    #        "baseline_50","baseline_100","baseline_150","baseline_200",
    #         "three_phase_baseline_50","three_phase_baseline_100","three_phase_baseline_150","three_phase_baseline_200",
    #         "plasticity_500_5e-06","plasticity_500_1e-06","plasticity_500_5e-07","plasticity_500_1e-07","plasticity_500_0",
    #         "plasticity_500_5e-06_prune_only","plasticity_500_1e-06_prune_only","plasticity_500_5e-07_prune_only","plasticity_500_1e-07_prune_only","plasticity_500_0_prune_only",
    #         "static_replay_500_5e-06","static_replay_500_1e-06","static_replay_500_5e-07","static_replay_500_1e-07","static_replay_500_0",
    #         # "nest_300_100_p0.1_refacc89_flooracc87",
    #         # "nest_300_100_p0.1_refacc89_flooracc87.5",
    #         # "nest_300_100_p0.1_refacc89_flooracc88",
    #         # "nest_300_100_p0.1_refacc89_flooracc88.5",
    #         # "nest_300_100_p0.1_refacc89_flooracc89",
    #         "nest_300_100_p0.1_refacc95_flooracc95",
    #         "nest_300_100_p0.1_refacc95_flooracc94.5",
    #         "nest_300_100_p0.1_refacc95_flooracc94",
    #         "nest_300_100_p0.1_refacc95_flooracc93.5",
    #         "nest_300_100_p0.1_refacc95_flooracc93",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa1",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa0.998",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa0.996",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa0.994",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa0.992",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa0.99",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     show_pareto_frontier=True,
    #     pareto_scope="per_junctures_mode",
    #     pareto_frontier_kinds=("baseline", "three_phase", "plasticity", "plasticity_three_phase","static_replay", "nest", "dropnet"),
    #     save_path_out="results_kmnist_acc.png",
    #     title="Test Acc vs Parameter Count",
    #     y_col="Test Acc",
    #     ylabel="Test Acc",
    # )


    # plot_param_count_vs_test_acc(
    #     save_path="results_kmnist_new",
    #     experiments=[
    #        "baseline_50","baseline_100","baseline_150","baseline_200",
    #         "three_phase_baseline_50","three_phase_baseline_100","three_phase_baseline_150","three_phase_baseline_200",
    #         "plasticity_500_5e-06","plasticity_500_1e-06","plasticity_500_5e-07","plasticity_500_1e-07","plasticity_500_0",
    #         "plasticity_500_5e-06_prune_only","plasticity_500_1e-06_prune_only","plasticity_500_5e-07_prune_only","plasticity_500_1e-07_prune_only","plasticity_500_0_prune_only",
    #         "static_replay_500_5e-06","static_replay_500_1e-06","static_replay_500_5e-07","static_replay_500_1e-07","static_replay_500_0",
    #         # "nest_300_100_p0.1_refacc89_flooracc87",
    #         # "nest_300_100_p0.1_refacc89_flooracc87.5",
    #         # "nest_300_100_p0.1_refacc89_flooracc88",
    #         # "nest_300_100_p0.1_refacc89_flooracc88.5",
    #         # "nest_300_100_p0.1_refacc89_flooracc89",
    #         "nest_300_100_p0.1_refacc95_flooracc95",
    #         "nest_300_100_p0.1_refacc95_flooracc94.5",
    #         "nest_300_100_p0.1_refacc95_flooracc94",
    #         "nest_300_100_p0.1_refacc95_flooracc93.5",
    #         "nest_300_100_p0.1_refacc95_flooracc93",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa1",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa0.998",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa0.996",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa0.994",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa0.992",
    #         # "dropnet_300_100_p0.2_modeglobal_kappa0.99",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     show_pareto_frontier=True,
    #     pareto_scope="per_junctures_mode",
    #     pareto_frontier_kinds=("baseline", "three_phase", "plasticity", "plasticity_three_phase","static_replay", "nest", "dropnet"),
    #     save_path_out="results_kmnist_brier.png",
    #     title="Test Brier vs Parameter Count",
    #     y_col="Test Brier",
    #     ylabel="Test Brier",
    # )

    # plot_param_count(
    #     save_path="results_fashionmnist_new",
    #     experiments=[
    #         "plasticity_500_5e-06","plasticity_500_1e-06","plasticity_500_5e-07","plasticity_500_1e-07","plasticity_500_0",

    #     ],
    #     show_individual=False,   # mean ± std across run_1..run_5
    #     show_checkpoint=True,    # dot at selected checkpoint
    #     save_path_out="results_fashionmnist_param_count_vs_epoch.png",
    #     show=True,
    # )

    #     plot_structural_action_proportions(
    #     save_path="results_fashionmnist_new",
    #     experiments=[
    #         "plasticity_500_0",
    #         "plasticity_500_1e-07",
    #         "plasticity_500_5e-07",
    #         "plasticity_500_1e-06",
    #         "plasticity_500_5e-06",
    #     ],
    #     save_path_out="results_fashionmnist_decision_proportions.png",
    # )


    # plot_structural_decision_heatmap(
    #     save_path="results_kmnist_new",
    #     experiments=[
    #         "plasticity_500_0",
    #         "plasticity_500_1e-07",
    #         "plasticity_500_5e-07",
    #         "plasticity_500_1e-06",
    #         "plasticity_500_5e-06",
    #     ],
    #     save_path_out="results_kmnist_decision_heatmap.png",
    # )