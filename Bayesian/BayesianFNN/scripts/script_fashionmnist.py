from experiments import run_nest, run_plasticity, run_shift
from lib.seed import SEED, set_seed


if __name__ == "__main__":
    # plasticity 
    for hidden_size in [[500,500]]:
        for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
            for i in range(1,6):
                for junctures_mode in ["both"]:
                    set_seed(SEED+i)
                    run_plasticity.main(
                        f"results_fashionmnist/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="fashion_mnist",
                        run_mode="plasticity",
                        warm_start_steps=64,
                        grow_new_only_steps=16,
                        gamma=0.0,
                        random_growth=False,
                    )
    # static symmetric
    for hidden_size in [[20,20],[50,50],[100,100],[150,150],[200,200]]:
        for lambda_penalty in [0]:
            for i in range(1,6):
                for junctures_mode in ["both"]:
                    set_seed(SEED+i)
                    run_plasticity.main(
                        f"results_fashionmnist/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="fashion_mnist",
                        run_mode="baseline",
                    )

    # static replay
    for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
        for i in range(1, 6):
            set_seed(SEED + i)
            plasticity_dir = (
                f"results_fashionmnist/run_{i}/plasticity_500_{float(lambda_penalty):g}"
            )
            run_plasticity.main(
                f"results_fashionmnist/run_{i}",
                [500, 500],
                lambda_penalty=0,
                dataset="fashion_mnist",
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
                        f"results_fashionmnist/run_{i}",
                        hidden_size,
                        lambda_penalty,
                        junctures_mode=junctures_mode,
                        dataset="fashion_mnist",
                        run_mode="plasticity",
                        warm_start_steps=64,
                        grow_new_only_steps=16,
                        gamma=0.0,
                    )


    # three-phase 
    for hidden_size in [[20, 20],[50, 50],[100, 100],[150, 150],[200, 200]]:
        for i in range(1, 6):
            set_seed(SEED + i)
            run_plasticity.main(
                f"results_fashionmnist/run_{i}",
                hidden_size,
                lambda_penalty=0,
                dataset="fashion_mnist",
                run_mode="three_phase",
                phase1_epochs=10,
                phase2_epochs=10,
                phase3_epochs=10,
                three_phase_growth_layer_idx=1,
                three_phase_growth_gamma=1,
            )

    #NeST-inspired
    for prune_acc_floor in [89.0, 88.5, 88.0, 87.5, 87.0]:
        for i in range(1, 6):
            set_seed(SEED + i)
            run_nest.main(
                f"results_fashionmnist/run_{i}",
                hidden_sizes=(300, 100),
                dataset="fashion_mnist",
                seed_scale=1,
                seed_activate_frac=0.1,
                reference_acc=89.0,
                prune_acc_floor=prune_acc_floor,
                max_growth_epochs=60,
                grow_interval=2,
                conn_grow_frac=0.01,
                beta_growth=0.4,
                birth_strength=0.4,
                prune_frac=0.01,
                max_prune_rounds=200,
                prune_retrain_epochs=2,
            )

    # prior shift plasticity
    for hidden_size in [[500, 500]]:
        for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
            for i in range(1, 6):
                set_seed(SEED + i)
                run_shift.main(
                    f"results_prior_shift_fashionmnist/run_{i}",
                    hidden_size,
                    lambda_penalty=lambda_penalty,
                    run_mode="plasticity",  # or "baseline" / "static_replay"
                    junctures_mode="both",
                    phase2_vcl_prior=True, 
                    warm_start_steps=64,
                    grow_new_only_steps=16,
                    gamma=0.0,
                )

    # prior shift static symmetric
    for hidden_size in [[20,20],[50,50],[100,100],[150,150],[200,200]]:
        for i in range(1, 6):
            set_seed(SEED + i)
            run_shift.main(
                f"results_prior_shift_fashionmnist/run_{i}",
                hidden_size,
                lambda_penalty=0,
                run_mode="baseline",
                phase2_vcl_prior=True,
            )

    # prior shift static replay
    for lambda_penalty in [0,1e-07,5e-07,1e-06,5e-06]:
        for i in range(1, 6):
            set_seed(SEED + i)
            run_shift.main(
                f"results_prior_shift_fashionmnist/run_{i}",
                [500, 500], 
                lambda_penalty=0,
                run_mode="static_replay",
                resume_from_plasticity_dir=(
                    f"results_prior_shift_fashionmnist/run_{i}/"
                    f"plasticity_500_{float(lambda_penalty):g}_vcl"
                ),
                phase2_vcl_prior=True,
            )

