from .igmc import IGMCModel

MODELS = {
    IGMCModel.code(): IGMCModel,
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
