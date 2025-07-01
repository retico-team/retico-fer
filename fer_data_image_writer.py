import cv2
import numpy as np

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
                # Convert DetectedObjectsIU to ImageIU and draw bounding boxes
                #
                # image: Image = iu.image
                #
                # output_iu = self.create_iu(iu)
                #
                # obj_type = iu.object_type
                # num_objs = min(self.num_obj_to_display, iu.num_objects)
                #
                # if obj_type == 'bb':
                #     valid_boxes = iu.payload
                #     for i in range(num_objs):
                #         box = valid_boxes[i]
                #         if box is not None:
                #             x1, y1, x2, y2 = box
                #             # Draw bounding box on the image
                #             img_draw = ImageDraw.Draw(image)
                #             img_draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                #             # put a label on the box
                #             img_draw.text((x1, y1), f'Object {i + 1}', fill='red')
                # elif obj_type == 'seg':
                #     valid_segs = iu.payload
                #     for i in range(num_objs):
                #         seg_mask = valid_segs[i]
                #         # seg_mask is expected to be a binary mask
                #         if seg_mask is not None:
                #             # Convert the mask to a PIL Image and apply it to the original with transparency
                #             seg_mask_image = Image.fromarray(seg_mask.astype('uint8') * 255)
                #             # blend the mask with the original image
                #             image = Image.composite(image, Image.new('RGB', image.size, (255, 0, 0)), seg_mask_image)
                #             # put a label on the image
                #             img_draw = ImageDraw.Draw(image)
                #             img_draw.text((10, 10 + i * 20), f'Segmented Object {i + 1}', fill='red')
                # else:
                #     print('Object type is invalid. Can\'t retrieve segmented object.')
                #     exit()
                #
                # output_iu.image = image
                # um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
                # self.append(um)

    def process_iu(self, iu):
        print(f"---IU: {iu}---")
        print(f"Emotion: {iu.emotion}, Valence: {iu.valence}, Arousal: {iu.arousal}")

        image = iu.grounded_in.image
        # Convert PIL image to OpenCV format (numpy array in BGR)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        text = f"Emotion: {iu.emotion.capitalize()}, Valence: {iu.valence:.2f}, Arousal: {iu.arousal:.2f}"
        cv2.putText(image_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        output_iu: ImageIU = iu.grounded_in
        return UpdateMessage.from_iu(output_iu, UpdateType.ADD)