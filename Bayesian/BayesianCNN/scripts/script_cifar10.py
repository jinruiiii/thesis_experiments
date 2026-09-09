from experiments import run_plasticity
from experiments import run_nest
from lib.seed import SEED, set_seed


if __name__ == "__main__":

    # plasticity
    for channels in [[300,300]]:
        for lambda_penalty in [1e-07,5e-07,1e-06,2.5e-06,5e-06]:
            for i in range(1, 6):
                set_seed(SEED + i)
                run_plasticity.main(
                    save_path=f"./results_cifar10/run_{i}",
                    conv_channels=channels,
                    lambda_penalty=lambda_penalty,
                    run_mode="plasticity",
                    dataset="cifar10",
                    cifar10_grayscale=False,
                    fc_hidden=128,
                    warm_start_steps=32,
                    grow_new_only_steps=16
                )


    # static symmetric
    for channels in [[30,30],[40,40],[60,60],[80,80],[100,100]]:
        for lambda_penalty in [0]:
            for i in range(1, 6):
                set_seed(SEED + i)
                run_plasticity.main(
                    save_path=f"./results_cifar10/run_{i}",
                    conv_channels=channels,
                    lambda_penalty=lambda_penalty,
                    run_mode="baseline",
                    dataset="cifar10",
                    cifar10_grayscale=False,
                    fc_hidden=128,
                )


    # static replay
    plasticity_runs = [
        "plasticity_300f_300f_1e-07",
        "plasticity_300f_300f_5e-07",
        "plasticity_300f_300f_1e-06",
        "plasticity_300f_300f_2.5e-06",
        "plasticity_300f_300f_5e-06",
    ]

    for plasticity_name in plasticity_runs:
        for i in range(1, 6):
            set_seed(SEED + i)
            run_plasticity.main(
                save_path=f"./results_cifar10/run_{i}",
                conv_channels=[300, 300], 
                lambda_penalty=0,
                run_mode="static_replay",
                resume_from_plasticity_dir=(
                    f"./results_cifar10/run_{i}/{plasticity_name}"
                ),
                dataset="cifar10",
                cifar10_grayscale=False,
                fc_hidden=128,
            )

    # prune-only
    for channels in [[300,300]]:
        for lambda_penalty in [1e-07,5e-07,1e-06,2.5e-06,5e-06]:
            for i in range(1, 6):
                set_seed(SEED + i)
                run_plasticity.main(
                    save_path=f"./results_cifar10/run_{i}",
                    conv_channels=channels,
                    lambda_penalty=lambda_penalty,
                    run_mode="plasticity",
                    dataset="cifar10",
                    cifar10_grayscale=False,
                    fc_hidden=128,
                    warm_start_steps=32,
                    grow_new_only_steps=16,
                    junctures_mode="prune"
                )

    # three-phase
    for channels in [[30,30],[40,40],[60,60],[80,80],[100,100]]:
        for lambda_penalty in [0]:
            for i in range(1, 6):
                set_seed(SEED + i)
                run_plasticity.main(
                    save_path=f"./results_cifar10/run_{i}",
                    conv_channels=channels,
                    lambda_penalty=lambda_penalty,
                    run_mode="three_phase",
                    dataset="cifar10",
                    cifar10_grayscale=False,
                    fc_hidden=128,
                    phase1_epochs=20,
                    phase2_epochs=20,
                    phase3_epochs=50,
                    three_phase_growth_layer_idx=1,
                    three_phase_growth_gamma=1,
                )

    # NeST-inspired
    for prune_acc_floor in [71.5,71.25,71.0,70.75,70.5]:
        for i in range(1, 6):
            set_seed(SEED + i)
            run_nest.main(
                save_path=f"./results_cifar10/run_{i}",
                conv_channels=(300, 100),
                dataset="cifar10",
                cifar10_grayscale=False,
                seed_activate_frac=0.1,
                reference_acc=71.5,
                prune_acc_floor=prune_acc_floor,
                max_growth_epochs=200,
                grow_interval=2,
                conn_grow_frac=0.01,
                prune_frac=0.01,
                max_prune_rounds=500,
                prune_retrain_epochs=2,
            )

    # plasticity (random growth)
    for channels in [[300,300]]:
        for lambda_penalty in [1e-07,5e-07,1e-06,2.5e-06,5e-06]:
            for i in range(1, 6):
                set_seed(SEED + i)
                run_plasticity.main(
                    save_path=f"./results_cifar10/run_{i}",
                    conv_channels=channels,
                    lambda_penalty=lambda_penalty,
                    run_mode="plasticity",
                    dataset="cifar10",
                    cifar10_grayscale=False,
                    fc_hidden=128,
                    warm_start_steps=32,
                    grow_new_only_steps=16,
                    random_growth=True
                )
