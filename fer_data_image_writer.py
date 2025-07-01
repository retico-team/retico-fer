import cv2
import numpy as np
from PIL import ImageFont

from fer_output_iu import FEROutputIU
from retico_vision.vision import ImageIU, DetectedObjectsIU
from retico_core.abstract import AbstractModule
from retico_core import UpdateMessage, UpdateType

from PIL import Image, ImageDraw


class FERDataImageWriterModule(AbstractModule):
    """
    A module that writes FER data on its frame as an ImageIU.
    """

    @staticmethod
    def name():
        return "FERDataImageWriterModule"

    @staticmethod
    def description():
        return "A module that writes FER data on its frame as an ImageIU."

    @staticmethod
    def input_ius():
        return [FEROutputIU]

    @staticmethod
    def output_iu():
        return ImageIU

    def __init__(self):
        super().__init__()

    def process_update(self, update_message):
        for iu, um in update_message:
            if um == UpdateType.ADD:
                print(f"---Grounded id IU: {iu.grounded_in}---")
                return self.process_iu(iu)


    def process_iu(self, iu):
        print(f"---IU: {iu}---")
        print(f"Emotion: {iu.emotion}, Valence: {iu.valence}, Arousal: {iu.arousal}")

        image = iu.grounded_in.image

        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype("arial.ttf", size=20)
        if iu.emotion:
            draw.text((10, 10), f"Emotion: {iu.emotion.capitalize()}", fill="white", font=font)
        else:
            draw.text((10, 10), "Face not detected", fill="white", font=font)
        #
        # # Convert PIL image to OpenCV format (numpy array in BGR)
        # image_np = np.array(image)
        # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        #
        # text = f"Emotion: {iu.emotion.capitalize()}, Valence: {iu.valence:.2f}, Arousal: {iu.arousal:.2f}"
        # image = cv2.putText(image_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        output_iu: ImageIU = self.create_iu(iu)
        output_iu.set_image(image, iu.grounded_in.nframes, iu.grounded_in.rate)
        return UpdateMessage.from_iu(output_iu, UpdateType.ADD)