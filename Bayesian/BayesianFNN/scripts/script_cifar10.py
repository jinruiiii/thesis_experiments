from experiments import run_nest, run_plasticity, run_shift
from lib.seed import SEED, set_seed


if __name__ == "__main__":
    plasticity 
    for hidden_size in [[500,500]]:
        for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
            for i in range(1,6):
                for junctures_mode in ["both"]:
                    set_seed(SEED+i)
                    run_plasticity.main(
                        f"results_cifar10/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="cifar10",
                        run_mode="plasticity",
                        warm_start_steps=64,
                        grow_new_only_steps=32,
                        gamma=0.0,
                        random_growth=False,
                    )
    # static symmetric
    for hidden_size in [[30,30],[50,50],[100,100],[150,150],[200,200]]:
        for lambda_penalty in [0]:
            for i in range(1,6):
                for junctures_mode in ["both"]:
                    set_seed(SEED+i)
                    run_plasticity.main(
                        f"results_cifar10/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="cifar10",
                        run_mode="baseline",
                    )

    # static replay
    for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
        for i in range(1, 6):
            set_seed(SEED + i)
            plasticity_dir = (
                f"results_cifar10/run_{i}/plasticity_500_{float(lambda_penalty):g}"
            )
            run_plasticity.main(
                f"results_cifar10/run_{i}",
                [500, 500],
                lambda_penalty=0,
                dataset="cifar10",
                run_mode="static_replay",
                resume_from_plasticity_dir=plasticity_dir,
            )

    # prune-only 
    for hidden_size in [[500,500]]:
        for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
            for i in range(1,6):
                for junctures_mode in ["prune"]:
                    set_seed(SEED+i)
                    run_plasticity.main(
                        f"results_cifar10/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="cifar10",
                        run_mode="plasticity",
                        warm_start_steps=64,
                        grow_new_only_steps=32,
                        gamma=0.0,
                    )


    # three-phase 
    for hidden_size in [[30,30],[50,50],[100,100],[150,150],[200,200]]:
        for i in range(1, 6):
            set_seed(SEED + i)
            run_plasticity.main(
                f"results_cifar10/run_{i}",
                hidden_size,
                lambda_penalty=0,
                dataset="cifar10",
                run_mode="three_phase",
                three_phase_growth_layer_idx=1,
                three_phase_growth_gamma=1,
            )

    #NeST-inspired
    for prune_acc_floor in [44.0, 43.5, 43.0, 42.5, 42.0]:
        for i in range(1, 6):
            set_seed(SEED + i)
            run_nest.main(
                f"results_cifar10/run_{i}",
                hidden_sizes=(300, 100),
                dataset="cifar10",
                seed_scale=1,
                seed_activate_frac=0.1,
                reference_acc=44.0,
                prune_acc_floor=prune_acc_floor,
                max_growth_epochs=60,
                grow_interval=2,
                conn_grow_frac=0.01,
                beta_growth=0.4,
                birth_strength=0.4,
                prune_frac=0.01,
                max_prune_rounds=500,
                prune_retrain_epochs=2,
            )
