import subprocess

datasets = ["Aggregation", "S1", "R15"]

for dataset in datasets:
    print(f"Starting AE-GAN training with GMM for {dataset}...")
    
    if dataset == "S1" or dataset == "R15":
        command = [
        "python", "AE_GAN_two_GMM.py",
        "--dataset_name", dataset,
        "--n_classes", "15",
        "--n_epochs", "500",
        "--lr", "0.0004",
        "--b1", "0.5",
        "--b2", "0.99"
    ]
    elif dataset == "Aggregation":
        command = [
        "python", "AE_GAN_two_GMM.py",
        "--dataset_name", dataset,
        "--n_classes", "5",
        "--n_epochs", "500",
        "--lr", "0.0004",
        "--b1", "0.5",
        "--b2", "0.99"
    ]
    else:
        print(f"Unknown dataset: {dataset}")
        continue
    
    process = subprocess.run(command)
    
    if process.returncode == 0:
        print(f"Training completed for {dataset}")
    else:
        print(f"Error occurred while training on {dataset}")

print("All dataset training runs completed!")
print()
print()
print()
