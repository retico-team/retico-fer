from retico_core.abstract import IncrementalUnit

class FEROutputIU(IncrementalUnit):
    """A Dialogue Manager Decision.

    This IU represents the output for the Facial Expression Recognition process.

    Attributes:
        emotion (string): The estimated emotion by the FER model.
        valence (string): The computed emotional valence.
        arousal (string): The computed emotional arousal.
    """

    @staticmethod
    def type():
        return "Facial Expression Recognition output as Incremental Unit"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                 payload=None, emotion=None, valence=0.0, arousal=0.0, **kwargs):
        """Initialize the FerOutput.
        """
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=payload)
        self.emotion = emotion
        self.valence = valence
        self.arousal = arousal

    def set_output_fields(self, emotion, valence, arousal):
        self.emotion = emotion
        self.valence = valence
        self.arousal = arousal
        
    def __repr__(self):
        return f"FEROutputIU(emotion={self.emotion}, valence={self.valence}, arousal={self.arousal})"

