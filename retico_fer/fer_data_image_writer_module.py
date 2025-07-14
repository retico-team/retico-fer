from PIL import ImageFont

from fer_output_iu import FEROutputIU
from retico_vision.vision import ImageIU
from retico_core.abstract import AbstractModule
from retico_core import UpdateMessage, UpdateType

from PIL import ImageDraw


class FERDataImageWriterModule(AbstractModule):
    """
    FERDataImageWriterModule
    ------------------------

    A module that processes `FEROutputIU` instances and writes facial expression recognition (FER) data (emotion, valence, arousal)
    onto the corresponding image frames.
    The output is an `ImageIU` instance containing the annotated image.

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
                return self.process_iu(iu)


    def process_iu(self, iu):
        image = iu.grounded_in.image
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype("arial.ttf", size=20)
        if iu.emotion:
            draw.text((10, 10), f"Emotion: {iu.emotion.capitalize()}  Valence: {round(iu.valence, 2)}  Arousal: {round(iu.arousal, 2)}",
                      fill="white", font=font)
        else:
            draw.text((10, 10), "Face not detected", fill="white", font=font)

        output_iu: ImageIU = self.create_iu(iu)
        output_iu.set_image(image, iu.grounded_in.nframes, iu.grounded_in.rate)
        return UpdateMessage.from_iu(output_iu, UpdateType.ADD)