from retico_core.debug import DebugModule
import retico_core.network as network
from retico_screen import ScreenModule
from retico_vision import WebcamModule
from fer_module import FERModule
from fer_data_image_writer import FERDataImageWriterModule


msg = []

webcam = WebcamModule(640, 480)
fer = FERModule()
fer_data_image_writer = FERDataImageWriterModule()
screen = ScreenModule()
# debug = DebugModule()

webcam.subscribe(fer)
fer.subscribe(fer_data_image_writer)
fer_data_image_writer.subscribe(screen)

network.run(webcam)
print("Running the webcam")

input()

network.stop(webcam)

