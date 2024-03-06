def create_model(opt):
    from .HAT_model import HATGAN, InferenceModel
    if opt.isTrain:
        model = HATGAN()
    else:
        model = InferenceModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    return model
