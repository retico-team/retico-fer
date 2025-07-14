from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import dlib
from collections import deque
import threading
from emonet.models import EmoNet

from retico_core import AbstractModule
from retico_core import UpdateMessage
from retico_core.abstract import UpdateType
from retico_vision import ImageIU
from fer_output_iu import FEROutputIU


class FERModule(AbstractModule):
    """
    FERModule
    ---------

    A module for Facial Expression Recognition (FER) that processes input Incremental Units (IUs) containing images and outputs IUs with emotion data.

    This module uses a pre-trained EmoNet model to detect facial expressions, valence, and arousal from images. It is designed to work within the Retico framework.

    Attributes:
    -----------
    - emotion_class : str
        Specifies the set of emotion classes to use. Options are "basic" or "extended".
    - image_size : int
        The size to which input images are resized for processing.
    - emotion_class_sets : dict
        A dictionary containing mappings of emotion class indices to their corresponding labels.
    - emotion_classes : dict
        The selected emotion class set based on `emotion_class`.
    - device : str
        The device used for computation ("cuda" for GPU or "cpu").
    - face_detector : dlib.fhog_object_detector
        The face detection model used to locate faces in images.
    - fer_model : EmoNet
        The pre-trained EmoNet model used for emotion recognition.
    - emotion : str
        The detected emotion from the last processed image.
    - valence : float
        The computed emotional valence from the last processed image.
    - arousal : float
        The computed emotional arousal from the last processed image.
    - fer_thread : threading.Thread
        The thread responsible for processing FER data.
    - running : bool
        Indicates whether the FER processing loop is running.
    - queue : collections.deque
        A queue for storing input IUs to be processed.

    """
    @staticmethod
    def name():
        return "FERModule"

    @staticmethod
    def description():
        return "A Facial Expression Recognition module that processes input IUs and outputs text IUs with emotion data."

    @staticmethod
    def input_ius():
        return [ImageIU]

    @staticmethod
    def output_iu():
        return FEROutputIU

    def __init__(self, emotions_class="basic", **kwargs):
        super().__init__(**kwargs)
        self._emotion_class = emotions_class

        self._image_size = 256
        self._emotion_class_sets = {
            "basic": {
                0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear"
            },
            "extended": {
                0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear",
                5: "Disgust", 6: "Anger", 7: "Contempt"
            }
        }
        self._emotion_classes = self._emotion_class_sets[self._emotion_class]

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._face_detector = dlib.get_frontal_face_detector()
        self._fer_model = self.load_model()

        self._emotion = None
        self._valence = 0.0
        self._arousal = 0.0

        self._fer_thread = None
        self._running = False
        self._queue = deque(maxlen=1)

    def setup(self):
        self._running = True

        # Start the FER consuming loop thread
        self._fer_thread = threading.Thread(target=self.fer_consuming_loop, daemon=True)
        self._fer_thread.start()

    def load_model(self):
        model_dir = Path(__file__).resolve().parents[2] / 'retico_fer' / 'emonet' / 'pretrained'
        emotions_class_size = len(self._emotion_classes.keys())
        state_dict_path = model_dir / f'emonet_{emotions_class_size}.pth'
        print(f'Loading the fer_model from {state_dict_path}.')

        state_dict = torch.load(str(state_dict_path), map_location=self._device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        net = EmoNet(n_expression=emotions_class_size).to(self._device)
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        return net

    def process_update(self, update_message):
        for iu, um in update_message:
            if um == UpdateType.ADD:
                self._queue.append(iu)

    def fer_consuming_loop(self):
        while self._running:
            if len(self._queue) > 0:
                try:
                    input_iu = self._queue.popleft()
                    if input_iu is not None:
                        self.process_iu(input_iu)
                except Exception as e:
                    print(f"Error computing FER: {e}")

    def process_iu(self, input_iu):
        image = input_iu.image
        emotion, valence, arousal = self.get_facial_expression_data(face_image=image)
        self._emotion = emotion
        self._valence = valence
        self._arousal = arousal
        print(f"Emotion: {self._emotion}, Valence: {self._valence}, Arousal: {self._arousal}")

        output_iu: FEROutputIU = self.create_iu(input_iu)
        self.append(UpdateMessage.from_iu(output_iu, UpdateType.ADD))

    def get_facial_expression_data(self, face_image):
        image_tensor = self.preprocess_image(face_image)
        if image_tensor is None:
            return None, None, None

        with torch.no_grad():
            output = self._fer_model(image_tensor.unsqueeze(0))
            predicted_emotion_class = torch.argmax(nn.functional.softmax(output["expression"], dim=1)).cpu().item()
            valence = output['valence'].clamp(-1.0, 1.0).cpu().item()
            arousal = output['arousal'].clamp(-1.0, 1.0).cpu().item()
            return self._emotion_classes[predicted_emotion_class].lower(), valence, arousal

    def preprocess_image(self, image_rgb):
        image_rgb = np.array(image_rgb)
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._face_detector(gray_image)

        if len(faces) == 0:
            print("No face detected, skipping computation.")
            return None
        face = faces[0]
        resized = self.crop_image(face, image_rgb)

        image_tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image_tensor.to(self._device)

    def crop_image(self, face, image_rgb):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        x = max(0, x)
        y = max(0, y)
        w = min(image_rgb.shape[1] - x, w)
        h = min(image_rgb.shape[0] - y, h)

        cropped_face = image_rgb[y:y + h, x:x + w]
        if cropped_face.size == 0:
            print("Invalid cropped face, skipping computation.")
            return None

        return cv2.resize(cropped_face, (self._image_size, self._image_size))

    def create_iu(self, grounded_in=None):
        output_iu: FEROutputIU = super().create_iu(grounded_in=grounded_in)
        output_iu.set_output_fields(self._emotion, self._valence, self._arousal)
        return output_iu