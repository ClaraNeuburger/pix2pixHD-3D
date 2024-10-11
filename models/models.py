
def create_model(opt):
    model = None
    print(opt.model)

    elif opt.model == 'pix2pix3d':
        assert(opt.dataset_mode == 'nodule')
        from .pix2pixHD_model import Pix2Pix3dModel
        model = Pix2Pix3dModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model