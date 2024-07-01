def create_model(opt):
    from .HECL_model import HECL, InferenceModel
    if opt.isTrain:
        model = HECL()
    else:
        model = InferenceModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    return model
