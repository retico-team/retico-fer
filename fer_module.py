import cv2
import numpy as np
from pathlib import Path
import torch
from PIL.Image import Image
from torch import nn
import dlib
from collections import deque
import threading
from emonet.emonet.emonet.models import EmoNet

from retico_core import AbstractModule
from retico_core import UpdateMessage, UpdateType
from retico_core.abstract import UpdateType
from retico_vision import ImageIU
from fer_output_iu import FEROutputIU


class FERModule(AbstractModule):
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

    def __init__(self, emotions_set_size=5, **kwargs):
        super().__init__(**kwargs)
        self.previous_timestamp = None
        self.previous_decision = None
        self.n_classes = emotions_set_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_size = 256
        self.emotion_class_sets = {
            5: {
                0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear"
            },
            8: {
                0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear",
                5: "Disgust", 6: "Anger", 7: "Contempt"
            }
        }
        self.emotion_classes = self.emotion_class_sets[self.n_classes]
        self.model = self.load_model()
        self.face_detector = dlib.get_frontal_face_detector()

        self.emotion = None
        self.valence = 0.0
        self.arousal = 0.0
        self.fer_thread = None

        self._running = False
        self.queue = deque(maxlen=1)
        self.root = None
        self.label = None

    def setup(self):
        """
        Set up the module, e.g., initialize display settings.
        """
        self._running = True
        # Start the display thread
        self.fer_thread = threading.Thread(target=self._fer_loop, daemon=True)
        self.fer_thread.start()

    def load_model(self):
        model_dir = Path(__file__).resolve().parents[1] / 'retico_fer' / 'emonet' / 'emonet' / 'pretrained'
        state_dict_path = model_dir / f'emonet_{self.n_classes}.pth'
        print(f'Loading the model from {state_dict_path}.')

        state_dict = torch.load(str(state_dict_path), map_location=self.device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net = EmoNet(n_expression=self.n_classes).to(self.device)
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        return net

    def process_update(self, update_message):
        for iu, um in update_message:
            if um == UpdateType.ADD:
                self.queue.append(iu)

    def _fer_loop(self):
        while self._running:
            if len(self.queue) > 0:
                # try:
                input_iu = self.queue.popleft()
                if input_iu is not None:
                    self.process_iu(input_iu)
                # except Exception as e:
                #     print(f"Error computing FER: {e}")

    def process_iu(self, input_iu):
        print("Processing IU...")
        image = input_iu.image
        emotion, valence, arousal = self.get_face_emotion_data(face_image=image)
        self.emotion = emotion
        self.valence = valence
        self.arousal = arousal
        print(f"Emotion: {self.emotion}, Valence: {self.valence}, Arousal: {self.arousal}")

        output_iu : FEROutputIU = self.create_iu(input_iu)
        self.append(UpdateMessage.from_iu(output_iu, UpdateType.ADD))

    def get_face_emotion_data(self, face_image):
        image_tensor = self.preprocess_image(face_image)
        if image_tensor is None:
            return None, None, None

        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0))
            predicted_emotion_class = torch.argmax(nn.functional.softmax(output["expression"], dim=1)).cpu().item()
            valence = output['valence'].clamp(-1.0, 1.0).cpu().item()
            arousal = output['arousal'].clamp(-1.0, 1.0).cpu().item()
            return self.emotion_classes[predicted_emotion_class].lower(), valence, arousal

    def preprocess_image(self, image_rgb):
        image_rgb = np.array(image_rgb)
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector(gray_image)

        if len(faces) == 0:
            print("No face detected, skipping computation.")
            return None
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        x = max(0, x)
        y = max(0, y)
        w = min(image_rgb.shape[1] - x, w)
        h = min(image_rgb.shape[0] - y, h)

        cropped_face = image_rgb[y:y + h, x:x + w]
        if cropped_face.size == 0:
            print("Invalid cropped face, skipping computation.")
            return None

        resized = cv2.resize(cropped_face, (self.image_size, self.image_size))

        # cv2.imshow("Resized Face", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

        image_tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image_tensor.to(self.device)

    def create_iu(self, grounded_in=None):
        output_iu : FEROutputIU = super().create_iu(grounded_in=grounded_in)
        output_iu.set_output_fields(self.emotion, self.valence, self.arousal)
        return output_iu

    def show_image(self, image):
        # Convert PIL image to NumPy array
        image_np = np.array(image)

        # Convert RGB to BGR (OpenCV uses BGR format)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Display the image using OpenCV
        cv2.imshow("Webcam Emotion Recognition", image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




r = FERModule()
