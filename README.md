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
pip install certifi==2023.7.22 charset-normalizer==3.3.2 contourpy==1.0.7 \
cycler==0.11.0 filelock==3.13.1 fonttools==4.39.4 fsspec==2023.10.0 idna==3.4 \
importlib_resources==6.5.2 Jinja2==3.1.2 joblib==1.2.0 kiwisolver==1.4.4 \
llvmlite==0.40.1 MarkupSafe==2.1.3 matplotlib==3.7.1 mpmath==1.3.0 networkx==3.2.1 \
numba==0.57.0 numpy==1.24.3 packaging==23.1 pandas==2.0.2 Pillow==9.5.0 \
pyparsing==3.0.9 python-dateutil==2.8.2 pytz==2023.3 requests==2.31.0 \
scikit-learn==1.2.2 scipy==1.10.1 six==1.16.0 sympy==1.13.1 threadpoolctl==3.1.0 \
torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 typing_extensions==4.12.2 \
tzdata==2023.3 urllib3==2.0.7 zipp==3.21.0
```
```bash
pip install brotli unicodedata2
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


