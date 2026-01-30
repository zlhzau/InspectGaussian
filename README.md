# InspectGaussian

**InspectGaussian** is a comprehensive framework designed for the reconstruction and detection of large-scale orchards (e.g., citrus orchards). The framework integrates pose estimation, single-plant view extraction, and an improved **3D Gaussian Splatting (3DGS)** technique to achieve high-precision 3D reconstruction and analysis of orchard environments.

This repository currently open-sources the **Single Plant View Extraction** module. By utilizing RGB-D image sequences and corresponding camera poses, this module automatically extracts all observational views of each individual plant from continuous video streams through spatial projection and global ID matching algorithms.

---

## 🌟 Key Features

- **Automated Extraction**  
  Automatically segments large-scale scenes into individual plant datasets based on detection results and depth maps.

- **Spatial Consistency Matching**  
  Ensures ID stability across different views using 3D center-point distance and temporal conflict detection, with automatic merging of fragmented IDs.

- **Metadata Statistics**  
  Automatically generates runtime reports and GPU memory usage statistics upon completion for performance analysis.

- **Complete Output**  
  Exports cropped **Color**, **Depth**, and **Mask** data, along with independent camera trajectory files for each individual plant.

---

## 🛠 Prerequisites

Please ensure your environment has the following dependencies installed:

- **Python**: 3.8+
- **PyTorch**: CUDA support is recommended for YOLO-World acceleration
- **Open3D**: Used for point cloud transformation and coordinate processing
- **Ultralytics (YOLO-World)**: Used for zero-shot object detection
- **OpenCV**: Used for image I/O and processing
- **tqdm**: Progress bar display
- **NumPy**

---

## 🔧 Installation

```bash
pip install torch torchvision ultralytics open3d opencv-python tqdm numpy
