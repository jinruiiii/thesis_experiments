from experiments import run_plasticity
from experiments import run_nest
from lib.seed import SEED, set_seed


if __name__ == "__main__":

    # for channels in [[300,300]]:
    #     for lambda_penalty in [2.5e-06]:
    #         for i in range(1, 6):
    #             set_seed(SEED + i)
    #             run_plasticity.main(
    #                 save_path=f"./results_cifar10_plasticity_0.05_ws32_gn16/run_{i}",
    #                 conv_channels=channels,
    #                 lambda_penalty=lambda_penalty,
    #                 run_mode="plasticity",
    #                 dataset="cifar10",
    #                 cifar10_grayscale=False,
    #                 fc_hidden=128,
    #                 warm_start_steps=32,
    #                 grow_new_only_steps=16
    #             )

    # for channels in [[300,300]]:
    #     for lambda_penalty in [2.5e-06]:
    #         for i in range(1, 6):
    #             set_seed(SEED + i)
    #             run_plasticity.main(
    #                 save_path=f"./results_cifar10_plasticity_0.05_ws32_gn16/run_{i}",
    #                 conv_channels=channels,
    #                 lambda_penalty=lambda_penalty,
    #                 run_mode="plasticity",
    #                 dataset="cifar10",
    #                 cifar10_grayscale=False,
    #                 fc_hidden=128,
    #                 warm_start_steps=32,
    #                 grow_new_only_steps=16,
    #                 junctures_mode="prune"
    #             )

    # plasticity_runs = [
    #     "plasticity_300f_300f_1e-07",
    #     "plasticity_300f_300f_5e-07",
    #     "plasticity_300f_300f_1e-06",
    #     "plasticity_300f_300f_2.5e-06",
    #     "plasticity_300f_300f_5e-06",
    # ]

    # for plasticity_name in plasticity_runs:
    #     for i in range(1, 6):
    #         set_seed(SEED + i)
    #         run_plasticity.main(
    #             save_path=f"./results_cifar10_plasticity_0.05_ws32_gn16/run_{i}",
    #             conv_channels=[300, 300],  # ignored; read from plasticity summary
    #             lambda_penalty=0,
    #             run_mode="static_replay",
    #             resume_from_plasticity_dir=(
    #                 f"./results_cifar10_plasticity_0.05_ws32_gn16/run_{i}/{plasticity_name}"
    #             ),
    #             dataset="cifar10",
    #             cifar10_grayscale=False,
    #             fc_hidden=128,
    #         )

    # for channels in [[25,25],[100,100],[50,50],[75,75]]:
    #     for lambda_penalty in [0]:
    #         for i in range(1, 6):
    #             set_seed(SEED + i)
    #             run_plasticity.main(
    #                 save_path=f"./results_cifar10_plasticity_0.1_ws32_gn8/run_{i}",
    #                 conv_channels=channels,
    #                 lambda_penalty=lambda_penalty,
    #                 run_mode="baseline",
    #                 dataset="cifar10",
    #                 cifar10_grayscale=False,
    #                 fc_hidden=128,
    #             )

    # for channels in [[100,100],[80,80],[60,60],[40,40],[30,30]]:
    #     for lambda_penalty in [0]:
    #         for i in range(1, 6):
    #             set_seed(SEED + i)
    #             run_plasticity.main(
    #                 save_path=f"./results_cifar10_three_phase/run_{i}",
    #                 conv_channels=channels,
    #                 lambda_penalty=lambda_penalty,
    #                 run_mode="three_phase",
    #                 dataset="cifar10",
    #                 cifar10_grayscale=False,
    #                 fc_hidden=128,
    #                 phase1_epochs=20,
    #                 phase2_epochs=20,
    #                 phase3_epochs=50,
    #                 three_phase_growth_layer_idx=1,
    #                 three_phase_growth_gamma=1,
    #             )

# directory name is wrong is 0.1 and not 0.05 and also run the 0 lambda penalty experiment for plasticity

    for prune_acc_floor in [71.25, 70.75]:
        for i in range(1, 6):
            set_seed(SEED + i)
            run_nest.main(
                save_path=f"./results_cifar10_nest/run_{i}",
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