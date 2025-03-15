# Auto-Encoding-GAN
Paper: [Auto-Encoding Generative Adversarial Networks towards Mode Collapse Reduction and Feature Representation Enhancement](https://doi.org/10.3390/e25121657)
Paper GitHub Repo: https://github.com/luxiaoxiang0002/Auto-Encoding-GAN/tree/master

## 4.2. Experiments on Synthetic Datasets
Since the data and network size is relatively small for this section, I decided to run it locally. This project was originally designed for **Intel-based Windows/Linux machines** with **CUDA acceleration**. Since I am using a **MacBook Air (M3, 2024)**, I made the following key modifications:  

1. **Switched from CUDA to Metal (MPS)** – CUDA is not available on Apple Silicon, so I replaced CUDA-related operations with `torch.backends.mps`.  
2. **Adjusted DataLoader Parameters** – Reduced `num_workers` in `DataLoader` to improve compatibility with macOS.  
3. **Handled BatchNorm Compatibility** – Modified some `BatchNorm` operations to prevent issues with PyTorch MPS.  
4. **Ensured Proper Package Installation** – Installed `torch`, `torchvision`, and `torchaudio` using the **MPS backend**, and fixed package conflicts.  

---

### 🖥 System Information:
- Device: MacBook Air 2025
- Chip: Apple M3
- RAM: 8 GB
- OS: macOS Sonoma 14.6.1

### ⚡️ PyTorch & MPS Acceleration
- **Backend:** Metal (MPS)
- **CUDA:** Not Available
- **Check MPS Support(should be True):**  
  ```bash
  python -c "import torch; print(torch.backends.mps.is_available())"
  ```
  
### 🐍 **Python & Virtual Environment Setup**  
To reproduce this environment, follow these steps:  

1️⃣ **Create a Conda Virtual Environment**  
```bash
conda create -n research_env python=3.9
conda activate research_env
```
2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

## 4.3. Experiments on Image Datasets
TODO

## 📂 File Explanation

- **AE_GAN_two.py** – Example code for 2D dataset experiments.
- **AE_GAN_mnist.py** – Example code for the MNIST dataset experiments.
- **dataset/** – Contains the 2D datasets used in the paper.
- **Aggregation_images/**:
  - The mode coverage results of the generated data for the **Aggregation** 2D dataset.
  - Model files, classification images, learning curve plot
- **Aggregation_images_GMM/**:
  - The mode coverage results of the generated data for the **Aggregation** 2D dataset for AE-GAN with **GMM**.
  - Model files, classification images, learning curve plot
- **R15_images/**:
  - The mode coverage results of the generated data for the **R15** 2D dataset.
  - Model files, classification images, learning curve plot
- **R15_images_GMM/**:
  - The mode coverage results of the generated data for the **R15** 2D dataset for AE-GAN with **GMM**.
  - Model files, classification images, learning curve plot
-  **S1_images/**:
  - The mode coverage results of the generated data for the **S1** 2D dataset for AE-GAN with **GMM**.
  - Model files, classification images, learning curve plot
- **S1_images_GMM/**:
  - The mode coverage results of the generated data for the **S1** 2D dataset.
  - Model files, classification images, learning curve plot
- **mnist_images/**:
  - The mode coverage results of the generated data for the **MNIST** image dataset.
  - Model files.
- **requirements.txt** – List of Python packages required to run the code.


