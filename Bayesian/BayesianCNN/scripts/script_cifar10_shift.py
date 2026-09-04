from experiments import run_shift
from lib.seed import set_seed, SEED

if __name__ == "__main__":
    # for channels in [[100, 100], [30, 30], [80, 80], [60, 60], [40, 40]]:
    #     for i in range(1, 6):
    #         set_seed(SEED + i)
    #         run_shift.main(
    #             save_path=f"./results_prior_shift_cifar10/run_{i}",
    #             conv_channels=channels,
    #             run_mode="baseline",
    #             phase1_epochs=30,
    #             phase2_epochs=60,
    #             cifar10_grayscale=False,
    #             fc_hidden=128,
    #             phase2_vcl_prior=True, 
    #         )

    # for channels in [[300, 300]]:
    #     for lambda_penalty in [1e-07, 5e-07, 1e-06, 2.5e-06, 5e-06]:
    #         for i in range(1, 6):
    #             set_seed(SEED + i)
    #             run_shift.main(
    #                 save_path=f"./results_prior_shift_cifar10/run_{i}",
    #                 conv_channels=channels,
    #                 run_mode="plasticity",
    #                 phase1_epochs=30,
    #                 phase2_epochs=60,
    #                 lambda_penalty=lambda_penalty,
    #                 cifar10_grayscale=False,
    #                 fc_hidden=128,
    #                 phase2_vcl_prior=True,
    #                 warm_start_steps=32,
    #                 junctures_mode="both",
    #                 grow_new_only_steps=16,
    #             )

    plasticity_runs = [
        "plasticity_300f_300f_1e-07_vcl",
        "plasticity_300f_300f_5e-07_vcl",
        "plasticity_300f_300f_1e-06_vcl",
        "plasticity_300f_300f_2.5e-06_vcl",
        "plasticity_300f_300f_5e-06_vcl",
    ]

    for plasticity_name in plasticity_runs:
        for i in range(1, 6):
            set_seed(SEED + i)
            run_shift.main(
                save_path=f"./results_prior_shift_cifar10/run_{i}",  # mostly unused for output path
                conv_channels=[300, 300],  # ignored; read from summary
                run_mode="static_replay",
                resume_from_plasticity_dir=(
                    f"./results_prior_shift_cifar10/run_{i}/{plasticity_name}"
                ),
                phase1_epochs=30,
                phase2_epochs=60,
                cifar10_grayscale=False,
                fc_hidden=128,
                phase2_vcl_prior=True,
            )