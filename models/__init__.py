
from models.encoderOnlyComplex import encoderOnlyComplex, reload_encoderOnlyComplex
from models.encoderOnly1Complex import encoderOnly1Complex
from models.encoderOnly111Complex import encoderOnly111Complex
from models.encoderOnly2Complex import encoderOnly2Complex
from models.encoderOnly4Complex import encoderOnly4Complex
from models.encoderOnly5Complex import encoderOnly5Complex
from models.encoderOnly6Complex import encoderOnly6Complex
from models.encoderOnly7Complex import encoderOnly7Complex
from models.encoderOnly1falseComplex import encoderOnly1falseComplex
from models.encoderOnlyunetComplex import encoderOnlyunetComplex
from models.encoderOnlyBrainComplex import encoderOnlyBrainComplex
from models.encoderOnlyIXIComplex import encoderOnlyIXIComplex
from models.encoderOnlyACDCComplex import encoderOnlyACDCComplex
from models.SP_EOIR_ACDC import SP_EOIR_ACDC
from models.encoderOnlyDynamicComplex import encoderOnlyDynamicComplex
from models.LessNet import LessNet
from models.FourierNet import SYMNet
from models.FourierNet_ACDC import SYMNet as SYMNet_ACDC
from models.LKUNet import UNet as LKUNet
from models.voxelmorph import VxmDense as voxelMorph
from models.transmorph import TransMorph
from models.priorWarpComplex import priorWarpComplex
from models.LessNet_ACDC import UNet_ACDC
from models.LessNet_oasis import UNet as UNet_oasis
from models.encoderOnlyhalfBrain import encoderOnlyhalfBrain
from models.RDP import RDP
from models.VxmLKUnet2DComplex import VxmLKUnet2DComplex
from models.VxmLKUnetComplex import VxmLKUnetComplex
from models.encoderOnly2DComplex import encoderOnly2DComplex
from models.transmorph_acdc import TransMorph_ACDC
from models.memWarpComplex import memWarpComplex
from models.VxmLKUnetCardiacComplex import VxmLKUnetCardiacComplex
from models.SegReg import SegReg
#from models.DDIR import DDIR
from models.SAMIR import SAMIR
from models.FMIR import FMIR
from models.regdino0 import regdino
from models.regdino import regdino_mlp
from models.Deepatlas import Deepatlas

def getModel(opt):

    model_name = opt['model']
    nkwargs = opt['nkwargs']
    model = None

    if 'reload_encoderOnlyComplex' in model_name:
        model = reload_encoderOnlyComplex(**nkwargs)
    elif 'encoderOnlyComplex' in model_name:
        model = encoderOnlyComplex(**nkwargs)
    elif 'memWarpComplex' in model_name:
        model = memWarpComplex(**nkwargs)
    elif 'encoderOnly1Complex' in model_name:
        model = encoderOnly1Complex(**nkwargs)
    elif 'encoderOnly111Complex' in model_name:
        model = encoderOnly111Complex(**nkwargs)
    elif 'SegReg' in model_name:
        model = SegReg(inshape=opt['in_shape'])
    elif 'encoderOnly2Complex' in model_name:
        model = encoderOnly2Complex(**nkwargs)
    elif 'encoderOnly4Complex' in model_name:
        model = encoderOnly4Complex(**nkwargs)
    elif 'encoderOnly5Complex' in model_name:
        model = encoderOnly5Complex(**nkwargs)
    elif 'encoderOnly6Complex' in model_name:
        model = encoderOnly6Complex(**nkwargs)
    elif 'encoderOnly7Complex' in model_name:
        model = encoderOnly7Complex(**nkwargs)
    elif 'encoderOnly1falseComplex' in model_name:
        model = encoderOnly1falseComplex(**nkwargs)
    elif 'encoderOnlyunetComplex' in model_name:
        model = encoderOnlyunetComplex(**nkwargs)
    elif 'encoderOnly2DComplex' in model_name:
        model = encoderOnly2DComplex(**nkwargs)
    elif 'VxmLKUnet2DComplex' in model_name:
        model = VxmLKUnet2DComplex(**nkwargs)
    elif 'encoderOnlyIXIComplex' in model_name:
        model = encoderOnlyIXIComplex(**nkwargs)
    elif 'encoderOnlyBrainComplex' in model_name:
        model = encoderOnlyBrainComplex(**nkwargs)
    elif 'encoderOnlyhalfBrain' in model_name:
        model = encoderOnlyhalfBrain(**nkwargs)
    elif 'encoderOnlyACDCComplex' in model_name:
        model = encoderOnlyACDCComplex(**nkwargs)
    elif 'SP_EOIR_ACDC' in model_name:
        model = SP_EOIR_ACDC(**nkwargs)
    elif 'RDP' == model_name:
        model = RDP(inshape=opt['in_shape'])
    elif 'encoderOnlyDynamicComplex' in model_name:
        model = encoderOnlyDynamicComplex(**nkwargs)
    elif 'LessNet_ACDC' in model_name:
        model = UNet_ACDC(**nkwargs)
    elif 'LessNet_oasis' in model_name:
        model = UNet_oasis(**nkwargs)
    elif 'LessNet' in model_name:
        model = LessNet(**nkwargs)
    elif 'FourierNet_ACDC' in model_name:
        model = SYMNet_ACDC(**nkwargs)    
    elif 'FourierNet' in model_name:
        model = SYMNet(**nkwargs)
    elif 'VxmLKUnetComplex' in model_name:
        model = VxmLKUnetComplex(**nkwargs)
    elif 'VxmLKUnetCardiacComplex' in model_name:
        model = VxmLKUnetCardiacComplex(**nkwargs)
    elif 'LKUNet' in model_name:
        model = LKUNet(**nkwargs)
    elif 'voxelMorph' in model_name:
        model = voxelMorph(img_size=str(opt['in_shape']))
    elif 'Deepatlas' in model_name:
        model = Deepatlas(img_size=str(opt['in_shape']))
    elif 'SAMIR' in model_name:
        model = SAMIR(**nkwargs)
    elif 'FMIR' in model_name:
        model = FMIR(**nkwargs)
    elif 'regdino_mlp' in model_name:
        model = regdino_mlp(**nkwargs)
    elif 'regdino' in model_name:
        model = regdino(**nkwargs)
    #elif 'DDIR' in model_name:
    #    model = DDIR(**nkwargs)
    elif 'TransMorph_ACDC' == model_name:
        model = TransMorph_ACDC()
    elif 'transMorph' == model_name:
        model = TransMorph(img_size=str(opt['in_shape']))
    elif 'priorWarpComplex' in model_name:
        model = priorWarpComplex(**nkwargs)
    else:
        raise ValueError("Model %s not recognized." % model_name)

    model = model.cuda()
    print("----->>>> Model %s is built ..." % model_name)

    return model
