# 📄 Description

**medSeg** allows you to:

- Automatically process CT images and identify areas containing signs of tumors.
- Perform segmentation using deep learning models.
- Determine the coordinates of the center of detected tumors.
- Predict the presence of EGFR and KRAS mutations in CT samples using pre-trained models.
- Save processing results, including masks and segmented images, for further analysis.

The program features an intuitive graphical user interface based on **PyQt5**, allowing for easy data management and analysis result visualization.

---

## 🚀 Getting Started

### 📌 Requirements

Before installation, ensure you have Python **3.8+** installed along with the following dependencies:

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 🛠️ Dependencies

- `PyQt5==5.15.9`: For creating the user interface
- `torch==2.2.2`, `torchvision==0.17.2`: For processing deep learning models
- `opencv-python-headless==4.9.0.80`: For image processing and visualization
- `Pillow==10.3.0`: For working with image data
- `ultralytics==8.2.0`: For YOLO-based object detection models
- `numpy==1.26.4`: For numerical data processing

---

## 📂 Installation and Running

1. Clone the repository:

```bash
git clone https://github.com/<your-repository-name>.git
cd medSeg
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the application:

```bash
python main.py
```

---

## 🖥️ Instructions for Use

1. **"Browse" Button**
   - Opens a folder selection dialog containing the CT images
   - After selecting a folder, images will be available in the dropdown menu

2. **"Process" Button**
   - Initiates the analysis of images in the selected folder
   - The analysis results will create two new folders:
     - `detected/`: Contains images with detection overlays
     - `segmentated/`: Contains segmentation masks and results

3. **Selecting an Image from the Dropdown Menu**
   - After processing, select an image to view:
     - The original image
     - The tumor detection mask
     - Tumor center coordinates
     - Mutation analysis results for EGFR and KRAS

---

## ⚙️ Configuration Notes

### 🎮 GPU Acceleration
For GPU support, replace the torch installation in `requirements.txt` with:
```diff
-torch==2.2.2
+torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```
Then reinstall dependencies:
```bash
pip install -r requirements.txt
```

### 🍎 macOS Specific Setup
Install Qt5 dependencies via Homebrew:
```bash
brew install qt@5
export PATH="/opt/homebrew/opt/qt@5/bin:$PATH"
```

---

## 📊 Example Workflow

1. Select a directory containing CT images using "Browse"
2. Click "Process" to analyze all images
3. View results by selecting images from dropdown
4. Access processed files in:
   - `detected/`: Images with detection overlays
   - `segmentated/`: Segmentation masks and results
   - Tumor coordinates and mutation predictions displayed in GUI

---

## 🛡️ License

**medSeg** is licensed under the MIT License. You can use this code in personal and commercial projects. The license text must be retained in all uses.

---

## 🧠 Model Requirements
Place these model files in corresponding directories:
```
main_models/
  ├── det.pt
  └── seg.pt
mutation_models/
  ├── model_EGFR.pt
  └── model_KRAS.pt
```

## 🤝 Contributing & Citation

We welcome contributions! Please cite our work if using this software:

```bibtex
@article{Shariaty2025,
  title={AI-Driven Precision Oncology: 
         Integrating Deep Learning, Radiomics, and Genomic Analysis 
         for Enhanced Lung Cancer Diagnosis and Treatment},
  author={Shariaty, Faridoddin and Pavlov, Vitalii and Baranov, Maksim},
  journal={Signal, Image and Video Processing},
  year={2025},
  volume={19},
  pages={3285--3296},
  doi={10.1007/s11760-025-04244-y}
}
```

[![Springer Nature](https://img.shields.io/badge/Published%20in-Springer_Nature-%2310A5D3?style=flat&logo=springer)](https://link.springer.com/article/10.1007/s11760-025-04244-y)  
[📚 Read the full article on SpringerLink](https://link.springer.com/article/10.1007/s11760-025-04244-y)