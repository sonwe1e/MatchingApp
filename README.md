# Advanced Feature Matching Tool 🖼️✨

Welcome to the Advanced Feature Matching Tool! This application provides an interactive Streamlit interface for experimenting with and comparing various classical and deep learning-based feature detection and matching algorithms. Upload your own images, tweak parameters, and visualize results in real-time.

## 🚀 Description

This tool allows users to:

1.  **Upload two images** for feature matching.
2.  Select from a range of **feature detection algorithms**, including classic OpenCV methods (SIFT, ORB, AKAZE, BRISK) and modern deep learning models from Kornia (DISK, SuperPoint).
3.  Choose between different **feature matching algorithms**: OpenCV's Brute-Force Matcher (with ratio test/cross-check) and cutting-edge Kornia models (LightGlue, LoFTR).
4.  **Adjust algorithm-specific parameters** through an intuitive UI.
5.  **Visualize detected keypoints** on individual images.
6.  **Visualize matched features** between the two images.
7.  **Compare results** from different algorithm combinations side-by-side.

The application leverages PyTorch for deep learning model inference (running on CPU or GPU if available) and OpenCV for image processing and classical algorithms.

## ✨ Features

### Supported Feature Detectors:

- **OpenCV Classics**:
  - SIFT
  - ORB
  - AKAZE
  - BRISK
- **Kornia Deep Learning Models**:
  - DISK

### Supported Feature Matchers:

- **OpenCV**:
  - Brute-Force Matcher (BFMatcher)
    - Configurable with L1/L2 norms or Hamming norms based on detector.
    - Supports Lowe's Ratio Test.
    - Supports Cross-Checking.
- **Kornia Deep Learning Models**:
  - LightGlue (compatible with DISK and SuperPoint features)

### Key UI Functionalities:

- Interactive image uploads.
- Dynamic parameter sliders and input fields for selected algorithms.
- Real-time display of original images, keypoints, and matches.
- "Add to Comparison" feature to store and review results from different configurations.
- Device information (CPU/GPU) display with warnings for potentially slow operations on CPU.
- Option to clear comparison results.

## 🛠️ Tech Stack & Dependencies

- **Python 3.8+**
- **Streamlit**: For the web application interface.
- **OpenCV (cv2)**: For image processing, classical detectors/matchers, and drawing.
- **PyTorch**: For deep learning model inference (Kornia backend).
- **Kornia**: For deep learning-based feature detectors (DISK, SuperPoint) and matchers (LightGlue, LoFTR).
- **NumPy**: For numerical operations.
- **Pillow (PIL)**: For image loading.

All major dependencies are listed in `requirements.txt`.

## ⚙️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sonwe1e/MatchingApp.git
    cd MatchingApp
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _Note_: PyTorch will be installed according to your system's CUDA availability if possible (for GPU support) or as a CPU-only version. Ensure your environment meets PyTorch installation prerequisites if you intend to use a GPU.

## ▶️ Usage

Once the installation is complete, run the Streamlit application:

```bash
streamlit run app.py
```

This will typically open the application in your default web browser.

1.  Use the sidebar to upload two images.
2.  Select a feature detector and a feature matcher.
3.  Adjust their parameters as needed.
    - If using LoFTR as the matcher, the selected feature detector's output will be ignored, as LoFTR is an end-to-end method.
4.  Click "🚀 Compute & Add to Comparison".
5.  View the current keypoints and matches in the main area.
6.  Use the "Method Comparison Area" to select and view previously computed results.

## 📁 Project Structure

```
.
├── app.py                      # Main Streamlit application script
├── config.py                   # Application-wide configurations (constants, defaults)
├── requirements.txt            # Python dependencies
├── utils.py                    # Utility functions (image loading, drawing, etc.)
├── core/                       # Core computer vision logic
│   ├── __init__.py
│   ├── detectors.py            # Feature detection logic and model dispatching
│   ├── kornia_models.py        # Kornia model loading, preprocessing, and execution
│   └── matchers.py             # Feature matching logic and model dispatching
└── README.md                   # This file
```

## 🤝 Contributing

Contributions are welcome\! If you'd like to contribute, please feel free to fork the repository, make your changes, and submit a pull request.

Possible areas for contribution:

- Adding more feature detectors or matchers.
- Implementing advanced evaluation metrics (e.g., if ground truth is available).
- Improving UI/UX.
- Optimizing performance.
- Adding support for batch processing or video input.

Please ensure your code follows general Python best practices and includes appropriate comments and documentation.

Happy Matching\!
