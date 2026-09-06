from experiments import run_nest, run_plasticity
from lib.seed import SEED, set_seed


if __name__ == "__main__":


    # for hidden_size in [[500,500]]:
    #     for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
    #         for gamma in [0.01,0.05,0.1]:
    #             for i in range(1,6):
    #                 for junctures_mode in ["both"]:
    #                     set_seed(SEED+i)
    #                     print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
    #                     run_plasticity.main(
    #                         f"results_fashionmnist_plasticity_gamma{gamma}/run_{i}",
    #                         hidden_size,
    #                         lambda_penalty,
    #                         junctures_mode=junctures_mode,
    #                         dataset="fashion_mnist",
    #                         run_mode="plasticity",
    #                         warm_start_steps=64,
    #                         grow_new_only_steps=16,
    #                         gamma=gamma,
    #                     )

    # for hidden_size in [[30,30]]:
    #     for lambda_penalty in [0]:
    #             for i in range(1,6):
    #                 for junctures_mode in ["both"]:
    #                     set_seed(SEED+i)
    #                     print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
    #                     run_plasticity.main(
    #                         f"results_cifar10_plasticity_0.1_ws64_gn32/run_{i}",
    #                         hidden_size,
    #                         lambda_penalty,
    #                         junctures_mode=junctures_mode,
    #                         dataset="cifar10",
    #                         run_mode="baseline",
    #                         warm_start_steps=64,
    #                         grow_new_only_steps=32,
    #                     )


    # for prune_acc_floor in [42.5, 42.0]:
    #     for i in range(1, 6):
    #         set_seed(SEED + i)
    #         run_nest.main(
    #             f"results_cifar10_plasticity_0.1_ws64_gn32/run_{i}",
    #             hidden_sizes=(300, 100),
    #             dataset="cifar10",
    #             seed_scale=1,
    #             seed_activate_frac=0.1,
    #             reference_acc=44.0,
    #             prune_acc_floor=prune_acc_floor,
    #             max_growth_epochs=60,
    #             grow_interval=2,
    #             conn_grow_frac=0.01,
    #             beta_growth=0.4,
    #             birth_strength=0.4,
    #             prune_frac=0.01,
    #             max_prune_rounds=500,
    #             prune_retrain_epochs=2,
    #         )

    # print("All experiments completed.")

    for hidden_size in [[500,500]]:
        for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
            for i in range(1,6):
                for junctures_mode in ["both"]:
                    set_seed(SEED+i)
                    print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
                    run_plasticity.main(
                        f"results_fashionmnist_plasticity_random_growth/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="fashion_mnist",
                        run_mode="plasticity",
                        warm_start_steps=64,
                        grow_new_only_steps=16,
                        gamma=0.0,
                        random_growth=True,
                    )

    for hidden_size in [[500,500]]:
        for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
            for i in range(1,6):
                for junctures_mode in ["both"]:
                    set_seed(SEED+i)
                    print("Running experiment for run", i, f"(junctures_mode={junctures_mode})")
                    run_plasticity.main(
                        f"results_cifar10_plasticity_random_growth/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="cifar10",
                        run_mode="plasticity",
                        warm_start_steps=64,
                        grow_new_only_steps=32,
                        gamma=0.0,
                        random_growth=True,
                    )