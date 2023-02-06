from .mrpt import MRPT


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "mrpt":
        model = MRPT(
            config.MODEL.MRPT, ms_chans=config.MODEL.NUM_MS_BANDS, img_size=config.MODEL.PAN_SIZE, config=config
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
