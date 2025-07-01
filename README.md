### README

# Facial Expression Recognition Module (FERModule)

## Overview
This project implements a Facial Expression Recognition (FER) module using the `EmoNet` model, integrated with the `retico-core` and `retico-vision` frameworks. The module processes input images, detects faces, and predicts emotions along with valence and arousal values.

## Requirements
To run this project, the following dependencies are required:

### Python Libraries
- **EmoNet**: A pretrained deep learning model for emotion recognition.
  - Install EmoNet from its repository or ensure the `emonet` package is available in your environment.
- **retico-core**: A framework for incremental dialogue systems.
  - Install via pip: `pip install retico-core`
- **retico-vision**: A module for vision-based incremental units.
  - Install via pip: `pip install retico-vision`
- **OpenCV**: For image processing and visualization.
  - Install via pip: `pip install opencv-python`
- **NumPy**: For numerical operations.
  - Install via pip: `pip install numpy`
- **Torch**: For deep learning computations.
  - Install via pip: `pip install torch`
- **dlib**: For face detection.
  - Install via pip: `pip install dlib`

### Additional Requirements
- **Pretrained EmoNet Model**: Ensure the pretrained EmoNet model files (`emonet_5.pth` or `emonet_8.pth`) are located in the `retico_fer/emonet/emonet/pretrained` directory.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the pretrained EmoNet model files in the appropriate directory:
   ```
   retico_fer/emonet/emonet/pretrained/
   ```

## Usage
1. Run the FER module:
   ```bash
   python fer_module.py
   ```

2. The module will process input images, detect faces, and output emotion data.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- **EmoNet**: For providing the pretrained emotion recognition model.
- **retico-core** and **retico-vision**: For enabling incremental processing and vision-based modules.