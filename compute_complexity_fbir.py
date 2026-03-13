import time
import torch
import argparse
from torch import nn
from torchinfo import summary
from pytorch_memlab import profile
from ptflops import get_model_complexity_info
from utils import getters, setters
# from models.VxmTransBrainComplex import VxmTransBrainComplex
# from models.VxmDense import VxmDense
from models.voxelmorph import VxmDense as voxelMorph

'''
from models.VxmLKUnetSCPAbdomenComplex import VxmLKUnetSCPAbdomenComplex
from models.VxmLKUnetSCPAbdomenTMPComplex import VxmLKUnetSCPAbdomenTMPComplex
from models.VxmTransAbdomenComplex import VxmTransAbdomenComplex
from models.fourierNetAbdomenComplex import fourierNetAbdomenComplex


from models.backbones.lapirn_model import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3
'''

class SpatialTransformer(nn.Module):
    def __init__(self, size):
        super(SpatialTransformer, self).__init__()
        self.size = size
    def forward(self, x, flow):
        return x

class VecInt(nn.Module):
    def __init__(self, inshape, nsteps):
        super(VecInt, self).__init__()
        self.nsteps = nsteps
    def forward(self, vec):
        return vec

def setup_components(opt):
    upscale = [2, 2, 1]
    ss = opt['img_size']

    transformers = nn.ModuleList([
        SpatialTransformer((ss[0] // 2 ** i, ss[1] // 2 ** i, ss[2])).cuda()
        for i in range(5)  # 至少5个，因为模型需要索引4
    ])

    integrates = nn.ModuleList([
        VecInt((ss[0] // 2 ** i, ss[1] // 2 ** i, ss[2]), 7).cuda()
        for i in range(5)
    ])

    return transformers, integrates, upscale

def prepare_input(resolution):
    x1 = torch.FloatTensor(*resolution).cuda()
    x2 = torch.FloatTensor(*resolution).cuda()
    return dict(source=x1, target=x2)


def prepare_input1(resolution):
    x1 = torch.FloatTensor(*resolution).cuda()
    x2 = torch.FloatTensor(*resolution).cuda()
    resolution1 = (resolution[0], 3, resolution[2], resolution[3], resolution[4])
    x3 = torch.FloatTensor(*resolution1).cuda()
    return dict(x=x1, y=x2)


def prepare_input2(resolution):
    x1 = torch.FloatTensor(*resolution).cuda()
    x2 = torch.FloatTensor(*resolution).cuda()
    resolution[1] = 14
    x3 = torch.FloatTensor(*resolution).cuda()
    return dict(source=x1, target=x2, y_logits=x3)


def prepare_ones_input(resolution):
    """为 regdino_mlp 模型准备输入 - 257个通道"""
    batch_size, c, h, w, d = resolution

    # 基础图像通道 (1个通道)
    moving_base = torch.ones(batch_size, 1, h, w, d).cuda()
    fixed_base = torch.ones(batch_size, 1, h, w, d).cuda()

    # DINO特征通道 (256个通道)
    emb_dim = 256
    moving_emb = torch.ones(batch_size, emb_dim, h, w, d).cuda()
    fixed_emb = torch.ones(batch_size, emb_dim, h, w, d).cuda()

    # 拼接成257个通道
    moving = torch.cat([moving_base, moving_emb], dim=1)  # [1, 257, h, w, d]
    fixed = torch.cat([fixed_base, fixed_emb], dim=1)  # [1, 257, h, w, d]

    return moving, fixed

@profile
def compute_memory():
    x = prepare_input((batch_size, c, h, w, d))
    model = net.cuda()
    y = model(x['source'], x['target'])


def compute_time(net, moving, fixed, transformers, integrates, upscale, num):
    start_time = time.time()
    for i in range(num):
        _ = net(moving, fixed, transformers, integrates, upscale, registration=True)
    end_time = time.time()
    return (end_time - start_time) / num


def prepare_ones_input(resolution):
    """为 regdino_mlp 模型准备ones输入 - 257个通道"""
    batch_size, c, h, w, d = resolution

    # 基础图像通道 (1个通道)
    moving_base = torch.ones(batch_size, 1, h, w, d).cuda()
    fixed_base = torch.ones(batch_size, 1, h, w, d).cuda()

    # DINO特征通道 (256个通道)
    emb_dim = 256
    moving_emb = torch.ones(batch_size, emb_dim, h, w, d).cuda()
    fixed_emb = torch.ones(batch_size, emb_dim, h, w, d).cuda()

    # 拼接成257个通道
    moving = torch.cat([moving_base, moving_emb], dim=1)  # [1, 257, h, w, d]
    fixed = torch.cat([fixed_base, fixed_emb], dim=1)  # [1, 257, h, w, d]

    print(f"输入形状: moving={moving.shape}, fixed={fixed.shape}")
    return moving, fixed


def run_model(net, batch_size, c, h, w, d, opt, **kwargs):
    with torch.cuda.device(int(opt['gpu_id'])):
        transformers, integrates, upscale = setup_components(opt)
        model = net

        # 注意：这里传入的 c=1，但实际创建257个通道
        moving, fixed = prepare_ones_input((batch_size, 1, h, w, d))

        infer_time = compute_time(model.cuda(), moving, fixed, transformers, integrates, upscale, 3)
        print("infer_time:", infer_time)

        # 模型摘要 - 使用257个通道
        input_shape = (batch_size, 257, h, w, d)
        summary(model, input_size=[input_shape, input_shape])

        compute_memory(model, transformers, integrates, upscale)

        # FLOPs计算 - 使用257个通道
        macs, params = get_model_complexity_info(
            model,
            (batch_size, 257, h, w, d),  # 修改这里！
            input_constructor=prepare_input_for_regdino,  # 修改这里！
            as_strings=False,
            print_per_layer_stat=False
        )

        macs = macs / 1e9
        params = params / 1e6
        print('%s %.8f G' % ('FLOPs: ', macs * 2))
        print('%s %.8f G' % ('MACs: ', macs))
        print('%s %.8f M' % ('Parameters: ', params))


if __name__ == '__main__':
    opt = {
        'logs_path': './logs',
        'log': './logs',
        'save_freq': 5,
        'n_checkpoints': 2,
        'num_workers': 4,
        'in_shape': (128, 128, 16),
    }

    batch_size = 1
    c, h, w, d = 1, 128, 128, 16

    parser = argparse.ArgumentParser(description="cardiac")
    parser.add_argument("-m", "--model", type=str, default='regdino_mlp_rm257_car_un')
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-d", "--dataset", type=str, default='acdcreg')
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("-dp", "--datasets_path", type=str, default="./../../../data/")
    parser.add_argument("--load_ckpt", type=str, default="best")
    parser.add_argument("--field_split", type=str, default='test')
    parser.add_argument("--img_size", type=str, default='(128,128,16)')
    parser.add_argument("--fea_type", type=str, default='unet')

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]: s.split('=')[1] for s in unknowns}
    opt['img_size'] = eval(opt['img_size'])

    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)
    net, _ = getters.getTestModelWithCheckpoints(opt)

    run_model(net, batch_size, c, h, w, d, opt)
    '''
    python compute_complexity.py -m LessNet -d abdomenreg -bs 1 start_channel=32 --gpu_id 5
    python compute_complexity.py -m encoderOnlyComplexS32 -d acdcreg -bs 1 start_channel=32 --gpu_id 5
    CUDA_VISIBLE_DEVICES=6 python compute_complexity_acdc.py -m encoderOnlyACDCComplex8 start_channel=8 -d acdcreg -bs 1 --gpu_id 0
    '''