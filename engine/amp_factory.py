from effects.amp_model import AmpModel


class AmpFactory:
    """
    Creates and configures AmpModel instances by model name.
    """

    @staticmethod
    def create(model_name, sample_rate=44100):
        return AmpModel(
            sample_rate=sample_rate,
            model=model_name,
        )

    @staticmethod
    def configure(amp_model, model_name):
        amp_model.set_model(model_name)
        return amp_model