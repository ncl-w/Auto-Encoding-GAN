# Auto-Encoding-GAN

## Reproduction of 4.2. Experiments on Synthetic Datasets
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
- **Check MPS Support:**  
  ```bash
  python -c "import torch; print(torch.backends.mps.is_available())"
  ```




## README FROM ORIGINAL PAPER
1、本代码涉及的论文：  
Auto-Encoding Generative Adversarial Networks Towards Mode Collapse Reduction and Feature Representation Enhancement  

2、本代码需要运行的环境：  
处理器：13th Gen Intel(R) Core(TM) i7-13700H   2.40 GHz  
内存：16.0 GB (15.7 GB 可用)  
系统：	64 位操作系统, 基于 x64 的处理器  

3、本代码提供了二维数据集和图像数据集模mnist的样例代码。  
AE_GAN_two:二维数据集的样例代码。  
AE_GAN_mnist:二维数据集的样例代码。  
dataset:论文涉及的二维数据集。  
R15_images:文件中包括：二维数据集R15生成数据的模式覆盖结果和模型文件。  
mnist_images:文件中包括：图像数据集mnist生成数据的模式覆盖结果和模型文件。  
requirement.txt:代码运行需要安装的python包。


