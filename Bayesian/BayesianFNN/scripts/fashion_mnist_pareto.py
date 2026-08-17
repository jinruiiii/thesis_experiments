from experiments.run_plasticity import _format_lambda_dir, main
from lib.plot import get_statistics, plot_param_count, plot_param_count_vs_test_acc
from lib.seed import SEED, set_seed

if __name__ == "__main__":
    # hidden_sizes = [500, 500]
    # for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
    #     for junctures_mode in ["both"]:
    #         for i in range(1, 6):
    #             set_seed(SEED+i)
    #             print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
    #             main(
    #                 f"results_FashionMnist_FNN_ws864/run_{i}",
    #                 hidden_sizes,
    #                 lambda_penalty,
    #                 junctures_mode=junctures_mode,
    #                 dataset="fashion_mnist",
    #                 run_mode="plasticity",
    #             )
    # print("All FashionMnist experiments completed.")

    # hidden_sizes = [500, 500]
    # for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
    #     for junctures_mode in ["prune"]:
    #         for i in range(1, 6):
    #             set_seed(SEED+i)
    #             print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
    #             main(
    #                 f"results_FashionMnist_FNN_ws864/run_{i}",
    #                 hidden_sizes,
    #                 lambda_penalty,
    #                 junctures_mode=junctures_mode,
    #                 dataset="fashion_mnist",
    #                 run_mode="plasticity",
    #             )
    # print("All FashionMnist experiments completed.")

    # hidden_sizes = [500, 500]
    # for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
    #     for junctures_mode in ["prune"]:
    #         for i in range(1, 6):
    #             set_seed(SEED+i)
    #             print("Running Fashion experiment for run", i, f"(junctures_mode={junctures_mode})")
    #             main(
    #                 f"results_FashionMnist_FNN_new/run_{i}",
    #                 hidden_sizes,
    #                 lambda_penalty,
    #                 junctures_mode=junctures_mode,
    #                 dataset="fashion_mnist",
    #                 run_mode="plasticity",
    #             )
    # print("All FashionMnist experiments completed.")



    # for hidden_size in [[20,20],[50,50],[100,100],[150,150],[200,200]]:
    #     for lambda_penalty in [0]:
    #         for junctures_mode in ["both"]:
    #             for i in range(1,6):
    #                 set_seed(SEED+i)
    #                 print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
    #                 main(
    #                     f"results_FashionMnist_FNN_new/run_{i}",
    #                     hidden_size,
    #                     lambda_penalty,
    #                     junctures_mode=junctures_mode,
    #                     dataset="fashion_mnist",
    #                     run_mode="baseline",
    #                 )
    # print("All experiments completed.")



    for hidden_size in [[200,200],[150,150],[100,100],[50,50],[20,20]]:
        for lambda_penalty in [1e-06]:
            for junctures_mode in ["both"]:
                for i in range(1,6):
                    set_seed(SEED+i)
                    print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
                    main(
                        f"results_FashionMnist_FNN_ws1664/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="fashion_mnist",
                        run_mode="three_phase",
                        three_phase_growth_layer_idx=0,
                        three_phase_growth_gamma=1,
                    )
    print("All experiments completed.")


    # # Static replay: train a fixed FNN using Hidden Sizes from each plasticity run's summary.
    # for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
    #     for i in range(1, 6):
    #         set_seed(SEED+i)
    #         plasticity_dir = (
    #             f"results_FashionMnist_FNN_new/run_{i}/plasticity_500_{_format_lambda_dir(lambda_penalty)}"
    #         )
    #         print(
    #             "Running static replay for run",
    #             i,
    #             f"(source={plasticity_dir})",
    #         )
    #         main(
    #             f"results_FashionMnist_FNN_new/run_{i}",
    #             [500, 500],  # unused for architecture; taken from plasticity summary
    #             lambda_penalty=0,
    #             dataset="fashion_mnist",
    #             run_mode="static_replay",
    #             resume_from_plasticity_dir=plasticity_dir,
    #         )
    # print("All static replay experiments completed.")

    # hidden_sizes = [500, 500]
    # for lambda_penalty in [5e-06, 1e-06, 5e-07, 5e-08, 0]:
    #     for junctures_mode in ["prune"]:
    #         for i in range(1, 6):
    #             set_seed(SEED+i)
    #             print("Running FashionMnist experiment for run", i, f"(junctures_mode={junctures_mode})")
    #             main(
    #                 f"results_FashionMnist_FNN/run_{i}",
    #                 hidden_sizes,
    #                 lambda_penalty,
    #                 junctures_mode=junctures_mode,
    #                 dataset="fashion_mnist",
    #                 run_mode="plasticity",
    #             )
    # print("All FashionMnist experiments completed.")



    # hidden_sizes = [200,200]
    # for r in [1]:
    #     for lambda_penalty in [5e-06]:
    #         for junctures_mode in ["both"]:
    #             set_seed(SEED)
    #             for i in range(1,6):
    #                 print("Running experiment for run", i)
    #                 main(
    #                     f"results_FashionMnist_FNN/run_{i}",
    #                     [int(hidden_size * r) for hidden_size in hidden_sizes],
    #                     lambda_penalty,
    #                     phase1_epochs=100,
    #                     phase2_epochs=100,
    #                     phase3_epochs=100,
    #                     three_phase_growth_layer_idx=1,
    #                     three_phase_growth_gamma=1,
    #                     run_mode="three_phase",
    #                 )
    # print("All experiments completed.")


    # hidden_sizes = [200,200]
    # for r in [0.7]:
    #     for lambda_penalty in [1e-08]:
    #         for junctures_mode in ["both"]:
    #             set_seed(SEED)
    #             for i in range(1,6):
    #                 print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
    #                 main(
    #                     f"results_FashionMnist_FNN/run_{i}",
    #                     [int(hidden_size * r) for hidden_size in hidden_sizes],
    #                     lambda_penalty,
    #                     junctures_mode=junctures_mode,
    #                 )
#     plot_param_count(
#     save_path="results_temp_maske_mode_none_bidirectional_unc",
#     experiments=[
#         "plasticity_32_5e-07","plasticity_128_5e-07","plasticity_256_5e-07",
#     ],
# )

    # plot_param_count_vs_test_acc(
    #     save_path="results_Kmnist_FNN",
    #     experiments=[
    #        "baseline_20","baseline_50","baseline_100","baseline_150","baseline_200",
    #         "plasticity_500_5e-06","plasticity_500_1e-06","plasticity_500_5e-07","plasticity_500_5e-08","plasticity_500_0",
    #         # "static_replay_400_5e-06","static_replay_400_1e-06","static_replay_400_5e-07","static_replay_400_1e-07","static_replay_400_5e-08","static_replay_400_5e-09",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     show_pareto_frontier=True,
    #     pareto_frontier_kinds=("baseline", "plasticity", "static_replay"),
    #     save_path_out="results_FashionMnist_FNN_static_replay.png",
    #     title="Test Acc vs Parameter Count",
    #     y_col="Test Acc",
    #     ylabel="Test Acc",
    # )

    # plot_param_count_vs_test_acc(
    #     save_path="results_FashionMnist_FNN_new",
    #     experiments=[
    #        "baseline_20","baseline_50","baseline_100","baseline_150","baseline_200",
    #         "three_phase_baseline_20","three_phase_baseline_50","three_phase_baseline_100","three_phase_baseline_150","three_phase_baseline_200",
    #         "plasticity_500_5e-06","plasticity_500_1e-06","plasticity_500_5e-07","plasticity_500_1e-07","plasticity_500_0",
    #         "plasticity_500_5e-06_prune_only","plasticity_500_1e-06_prune_only","plasticity_500_5e-07_prune_only","plasticity_500_1e-07_prune_only","plasticity_500_0_prune_only",
    #         # "static_replay_500_5e-06","static_replay_500_1e-06","static_replay_500_5e-07","static_replay_500_1e-07","static_replay_500_0",
    #         # "nest_300_100_p0.1_refacc89_flooracc87",
    #         # "nest_300_100_p0.1_refacc89_flooracc87.5",
    #         # "nest_300_100_p0.1_refacc89_flooracc88",
    #         # "nest_300_100_p0.1_refacc89_flooracc88.5",
    #         # "nest_300_100_p0.1_refacc89_flooracc89",
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
    #     save_path_out="results_FashionMnist_FNN_acc.png",
    #     title="Test Acc vs Parameter Count",
    #     y_col="Test Acc",
    #     ylabel="Test Acc",
    # )


    # plot_param_count_vs_test_acc(
    #     save_path="results_FashionMnist_FNN_new",
    #     experiments=[
    #        "baseline_20","baseline_50","baseline_100","baseline_150","baseline_200",
    #         "three_phase_baseline_20","three_phase_baseline_50","three_phase_baseline_100","three_phase_baseline_150","three_phase_baseline_200",
    #         "plasticity_500_5e-06","plasticity_500_1e-06","plasticity_500_5e-07","plasticity_500_1e-07","plasticity_500_0",
    #         "plasticity_500_5e-06_prune_only","plasticity_500_1e-06_prune_only","plasticity_500_5e-07_prune_only","plasticity_500_1e-07_prune_only","plasticity_500_0_prune_only",
    #         "static_replay_500_5e-06","static_replay_500_1e-06","static_replay_500_5e-07","static_replay_500_1e-07","static_replay_500_0",
    #         # "nest_300_100_p0.1_refacc89_flooracc87",
    #         # "nest_300_100_p0.1_refacc89_flooracc87.5",
    #         # "nest_300_100_p0.1_refacc89_flooracc88",
    #         # "nest_300_100_p0.1_refacc89_flooracc88.5",
    #         # "nest_300_100_p0.1_refacc89_flooracc89",
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
    #     save_path_out="results_FashionMnist_FNN_brier.png",
    #     title="Test Brier vs Parameter Count",
    #     y_col="Test Brier",
    #     ylabel="Test Brier",
    # )

    # plot_param_count_vs_test_acc(
    #     save_path="results_FashionMnist_FNN3",
    #     experiments=[
    #         # "baseline_20", "baseline_50", "baseline_100", "baseline_150", "baseline_200",
    #         # small init
    #         "plasticity_20_5e-06", "plasticity_20_1e-06", "plasticity_20_5e-07", "plasticity_20_5e-08", "plasticity_20_0",
    #         # large init
    #         "plasticity_500_5e-06", "plasticity_500_1e-06", "plasticity_500_5e-07", "plasticity_500_5e-08", "plasticity_500_0",
    #     ],
    #     num_runs=5,
    #     aggregate_runs=True,
    #     # annotate_points=True,  # helps tell 20 vs 500 apart
    #     show_pareto_frontier=True,
    #     pareto_scope="per_init_width",
    #     pareto_frontier_kinds=("baseline", "plasticity"),
    #     save_path_out="results_FashionMnist_FNN.png",
    #     title="Test Brier vs Parameter Count",
    #     y_col="Test Brier",
    #     ylabel="Test Brier",
    # )


#make pruning and growing rates dynamic