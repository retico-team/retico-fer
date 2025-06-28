import sys, os

from retico_core.debug import DebugModule
import retico_core.network as network
from retico_vision import WebcamModule
from fer_module import FERModule


msg = []

webcam = WebcamModule(rate=20)
fer = FERModule()
debug = DebugModule()

webcam.subscribe(fer)
fer.subscribe(debug)

network.run(webcam)
print("Running the webcam")

input()

network.stop(webcam)

