from retico_core.abstract import IncrementalUnit

class FEROutputIU(IncrementalUnit):
    """
    FERModule
    ---------

    An Incremental Unit containing data from a Facial Expression Recognition module.

    Attributes:
    -----------
    - emotion : str
        The estimated emotion by the FER model.
    - valence : str
        The computed emotional valence.
    - arousal : str
        The computed emotional arousal.
    """

    @staticmethod
    def type():
        return "Facial Expression Recognition output as Incremental Unit"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None,
                 payload=None, emotion=None, valence=0.0, arousal=0.0, **kwargs):

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

