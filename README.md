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

Before installation, ensure you have Python **3.x** installed along with the following dependencies:

Install dependencies using `pip`:

```bash
pip install -r requrements.txt
```

### 🛠️ Dependencies

- `PyQt5`: For creating the user interface.
- `torch`, `torchvision`: For processing deep learning models.
- `cv2 (OpenCV)`: For image processing and visualization.
- `PIL`: For working with image data.
- `ultralytics`:For YOLO-based object detection models.
- `numpy`: For numerical data processing.

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

🖥️ Instructions for Use

1. "Browse" Button

- Opens a folder selection dialog containing the CT images.
- After selecting a folder, images will be available in the dropdown menu.

2. "Process" Button

- Initiates the analysis of images in the selected folder.
- The analysis results will create two new folders:

  - `detected/`: Contains images with detection overlays.
  - `segmentated/`: Contains segmentation masks and results.

3. Selecting an Image from the Dropdown Menu

- After processing, select an image from the dropdown list to view:

  - The original image.
  - The tumor detection mask.
  - Analysis results for mutations EGFR and KRAS.

4. Processing Results

Processing Output
After analysis, the results will be saved into the following folders:

- `segmentated/`: Segmented images from the tumor detection process.
- `detected/`: Images with overlays showing detected areas.

---

## 📊 Example Workflow

- Select a directory containing CT images using the "Browse" button.
- Click the "Process" button to analyze all images in the selected directory.
- Once processing is complete, select images from the dropdown menu to view the analysis results.
- Tumor Segmentation Masks: Saved as mask images.
- EGFR and KRAS Mutation Predictions: Displayed in the graphical user interface after processing.

---

## 🛡️ License

**medSeg** is licensed under the MIT License.

You can use this code in personal and commercial projects. However, the license text must be retained when used in any project.

---
