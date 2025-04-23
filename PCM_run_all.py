import subprocess
import time
from datetime import datetime

LOG_FILENAME = "PCM_run_all.log"
datasets = ["S1"]

def log_message(message):
    """
    Writes message to both standard output (print)
    and appends the same line to LOG_FILENAME.
    """
    print(message)
    with open(LOG_FILENAME, "a") as f:
        f.write(message + "\n")

for dataset in datasets:
    start_dt = datetime.now()
    start_ts = time.time()

    msg_start = f"Starting AE-GAN training with PCM for {dataset} at {start_dt}"
    log_message(msg_start)

    # Build the command for each dataset
    if dataset in ["S1", "R15"]:
        command = [
            "python", "AE_GAN_two_PCM.py",
            "--dataset_name", dataset,
            "--n_classes", "15",
            "--n_epochs", "500",
            "--lr", "0.0004",
            "--b1", "0.5",
            "--b2", "0.99",
            # If you need PCM hyperparams:
            # "--pcm_m", "2.0", "--pcm_e", "1e-4", "--pcm_max_iter", "50"
        ]
    elif dataset == "Aggregation":
        command = [
            "python", "AE_GAN_two_PCM.py",
            "--dataset_name", dataset,
            "--n_classes", "5",
            "--n_epochs", "500",
            "--lr", "0.0004",
            "--b1", "0.5",
            "--b2", "0.99",
            # e.g. "--pcm_m", "2.0", "--pcm_e", "1e-4", "--pcm_max_iter", "50"
        ]
    else:
        log_message(f"Unknown dataset: {dataset}")
        continue
    
    # Run the process
    process = subprocess.run(command)

    end_dt = datetime.now()
    end_ts = time.time()
    duration = end_ts - start_ts

    if process.returncode == 0:
        msg_ok = (f"Training completed for {dataset} at {end_dt}\n"
                  f"Duration (seconds): {duration:.2f}")
        log_message(msg_ok)
    else:
        msg_fail = (f"Error occurred while training on {dataset} at {end_dt}\n"
                    f"Duration before error (seconds): {duration:.2f}")
        log_message(msg_fail)

    log_message("-"*50)

log_message("All dataset training runs completed!")
log_message("")  # blank line at the end
