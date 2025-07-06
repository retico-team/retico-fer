# Retico Facial Expression Recognition Module

## Overview
This project implements a Facial Expression Recognition (FER) module using the `EmoNet` model, integrated with the `retico-core` and `retico-vision` frameworks. The module processes input images, detects faces, and predicts emotions along with valence and arousal values.

## Requirements
To run this project, the following dependencies are required:

### Python Libraries
- **EmoNet**: A pretrained deep learning model for emotion recognition.
  - Install EmoNet manually (see installation steps below).
- **retico-core**: A framework for incremental dialogue systems.
  - Install via pip: `pip install retico-core`
- **retico-vision**: A module for vision-based incremental units.
  - Install retico-vision manually (see installation steps below).
- **OpenCV**: For image processing and visualization.
  - Install via pip: `pip install opencv-python`
- **NumPy**: For numerical operations.
  - Install via pip: `pip install numpy`
- **Torch**: For deep learning computations.
  - Install via pip: `pip install torch`
- **Torchvision**: For image transformations and pretrained models.
  - Install via pip: `pip install torchvision`
- **dlib**: For face detection.
  - Install via pip: `pip install dlib`
- **Matplotlib**: For visualizing results.
  - Install via pip: `pip install matplotlib`
- **Scipy**: For scientific computations.
  - Install via pip: `pip install scipy`
- **Tqdm**: For progress bars.
  - Install via pip: `pip install tqdm`

### Additional Requirements
- **Pretrained EmoNet Model**: Ensure the pretrained EmoNet model files (`emonet_5.pth` or `emonet_8.pth`) are located in the `retico_fer/emonet/emonet/pretrained` directory.

## Installation

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/SimplyMarious/retico_fer.git
cd retico_fer
```

### Step 2: Install Required Python Libraries
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Install EmoNet Manually
Since the `EmoNet` repository does not include a `setup.py` or `pyproject.toml` file, it must be installed manually:
1. Clone the `EmoNet` repository:
   ```bash
   git clone https://github.com/face-analysis/emonet.git
   ```
2. Add the `emonet` directory to your `PYTHONPATH` environment variable using PowerShell:
   ```powershell
   $env:PYTHONPATH="$env:PYTHONPATH;C:\path\to\emonet"
   ```
   Replace `C:\path\to\emonet` with the actual path to the `emonet` directory.

### Step 4: Install retico-vision Manually
Since `retico-vision` must be installed manually, follow these steps:
1. Clone the `retico-vision` repository:
   ```bash
   git clone https://github.com/retico-team/retico-vision.git
   ```
2. Add the `retico-vision` directory to your `PYTHONPATH` environment variable using PowerShell:
   ```powershell
   $env:PYTHONPATH="$env:PYTHONPATH;C:\path\to\retico-vision"
   ```
   Replace `C:\path\to\retico-vision` with the actual path to the `retico-vision` directory.

### Step 5: Place Pretrained EmoNet Model Files
Download the pretrained EmoNet model files (`emonet_5.pth` or `emonet_8.pth`) and place them in the following directory:
```
retico_fer/emonet/emonet/pretrained/
```

## Usage
1. Run the FER module:
   ```bash
   python main.py
   ```

2. The module will process input images, detect faces, and output emotion data.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- **EmoNet**: For providing the pretrained emotion recognition model.
- **retico-core** and **retico-vision**: For enabling incremental processing and vision-based modules.
```