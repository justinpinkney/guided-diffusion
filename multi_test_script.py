import inference
import sys
from pathlib import Path


if __name__ == "__main__":
    checkpoint = Path(sys.argv[1])
    output_size = int(sys.argv[2])
    device = "cuda:0"
    cfg = Path("config.yaml")
    in_dir = Path("test_faces")
    batch_size = 2
    scale = 2

    timesteps = [
        "ddim50"
        "50",
        "ddim150",
        "150",
        "1000"
    ]


    for timestep in timesteps:

        out_dir = Path(f"{in_dir}_{timestep}_{output_size}")
        print(f"saving to {out_dir}")
        input_size = output_size//scale
        inference.main(
            checkpoint,
            config=cfg,
            input_dir=in_dir,
            output_dir=out_dir,
            output_size=output_size,
            input_size=input_size,
            timesteps=timestep,
            batch_size=batch_size,
            device=device,
        )
