# Retico Facial Expression Recognition Module

## Overview
This project implements a Facial Expression Recognition (FER) module using the `EmoNet` model, integrated with the `retico-core` and `retico-vision` frameworks. The module processes input images, detects faces, and predicts emotions along with valence and arousal values.

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

2. Add the `emonet` directory to your `PYTHONPATH` environment variable:

   **Using PowerShell**:
   ```powershell
   $env:PYTHONPATH="$env:PYTHONPATH;C:\path\to\emonet"
   ```
   Replace `C:\path\to\emonet` with the actual path to the `emonet` directory.

   **Using Bash**:
   ```bash
   export PYTHONPATH="$PYTHONPATH:/path/to/emonet"
   ```
   Replace `/path/to/emonet` with the actual path to the `emonet` directory.

### Step 4: Install retico-vision Manually
Since `retico-vision` must be installed manually, follow these steps:
1. Clone the `retico-vision` repository:
   ```bash
   git clone https://github.com/retico-team/retico-vision.git retico_vision
   ```
2. Add the `retico-vision` directory to your `PYTHONPATH` environment variable:

   **Using PowerShell**:
   ```powershell
   $env:PYTHONPATH="$env:PYTHONPATH;C:\path\to\retico_vision"
   ```
   Replace `C:\path\to\retico_vision` with the actual path to the `retico_vision` directory.

   **Using Bash**:
   ```bash
   export PYTHONPATH="$PYTHONPATH:/path/to/retico_vision"
   ```
   Replace `/path/to/retico_vision` with the actual path to the `retico_vision` directory.
### Step 5: Place Pretrained EmoNet Model Files
Download the pretrained EmoNet model files (`emonet_5.pth` or `emonet_8.pth`) and place them in the following directory:
```
retico_fer/emonet/emonet/pretrained/
```

## Usage
1. Enter the retico_fer directory and run the FER module:
   ```bash
   cd retico_fer
   python main.py
   ```

2. The module will process input images, detect faces, and output emotion data.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- **EmoNet**: For providing the pretrained emotion recognition model.
- **retico-core** and **retico-vision**: For enabling incremental processing and vision-based modules.
```