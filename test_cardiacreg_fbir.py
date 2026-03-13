import os
import torch
import interpol
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
from torch.autograd import Variable
import time

from utils import getters, setters
from utils.functions import AverageMeter, registerSTModel, dice_binary, dice_eval, computeJacDetVal, computeSDLogJ, \
    compute_HD95, jacobian_determinant

#########SAM
from models.segment_anything import build_sam, SamPredictor
from models.segment_anything import sam_model_registry
from models.segment_anything.modeling import Sam
from einops import repeat, rearrange

# SAM config
# interpolate = BilinearInterpolation(scale_factor=(0.5, 0.5, 1))
vit_name = 'vit_h'
sam_ckpt = 'sam_vit_h_4b8939.pth'
num_classes = 4
sam_size = 512  # self.inshape

sam = sam_model_registry[vit_name](image_size=sam_size,
                                   num_classes=num_classes,
                                   checkpoint=sam_ckpt, in_channel=3,
                                   pixel_mean=[0.5, 0.5, 0.5], pixel_std=[0.5, 0.5, 0.5]).cuda()
for p in sam.parameters():
    p.requires_grad = False
#########

###########dino
import torch.nn.functional as F
from models.backbones.voxelmorph.torch import layers
from transformers import AutoImageProcessor, AutoModel, Dinov2Config, AutoConfig
from einops import repeat, rearrange
from torchvision import transforms

transform_image = transforms.Compose([
    # transforms.Resize(image_size),
    # transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# backbone = torch.hub.load('/data/data4/hy/dinov3-vitb16-pretrain-lvd1689m','dinov3_vitb16', source='local', weights='model.safetensors')
# pretrained_model_name = "/data/data4/hy/dinov3-vitb16-pretrain-lvd1689m"
pretrained_model_name = "dinov3-vitb16-pretrain-lvd1689m"
'''
config = Dinov3Config.from_pretrained(pretrained_model_name)
config.image_size = 256  # 
config.patch_size = 16   #
'''
config = AutoConfig.from_pretrained(pretrained_model_name)
# print(config)
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
backbone = AutoModel.from_pretrained(pretrained_model_name, device_map="auto").cuda()
intermediate_layer_idx = {
    'small': [2, 5, 8, 11],
    'base': [2, 5, 8, 11],
}

encoder_size = 'base'
backbone = backbone

for p in backbone.parameters():
    p.requires_grad = False


###########

def embed_acdc(x, mode):
    B, C, H, W, D = x.shape
    x = torch.permute(x, (0, 4, 1, 2, 3)).view(-1, C, H, W)
    x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=3)
    x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
    x = transform_image(x)
    patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
    # print(x.shape)
    if mode == 'dino':
        outputs = backbone(x, output_hidden_states=True)
        intermediate_features = outputs.hidden_states
        #patch_features = outputs.last_hidden_state[:, 5:, :]
        patch_features = intermediate_features[2][:, 5:, :]
        feature_map = patch_features.reshape(-1, patch_h, patch_w, patch_features.shape[-1])
        feature_map = feature_map.permute(0, 3, 1, 2)  # [batch, dim, H, W]
    else:
        #
        feature_map = sam.image_encoder(x)
        # print(feature_map.shape)

    feature_map = torch.permute(feature_map, (1, 2, 3, 0)).unsqueeze(0)
    feature_map = F.interpolate(feature_map, scale_factor=(4, 4, 1), mode='trilinear', align_corners=True)
    # print(feature_map.shape)
    return feature_map


def embed_abdomen(x, mode):
    B, C, H, W, D = x.shape
    target_size = 128
    pad_h = (target_size - H) // 2
    pad_h_after = (target_size - H) // 2
    pad_w = (target_size - W) // 2
    pad_w_after = (target_size - W) // 2

    x = F.pad(x,
              # pad=(0, 0, pad_w, pad_w_after, pad_h, pad_h_after),  # W, H, D
              pad=(0, 0, pad_w, pad_w_after, pad_h, pad_h_after),
              mode='constant',
              value=0).cuda()
    # print(x.shape)#96, 80, 128
    x = torch.permute(x, (0, 4, 1, 2, 3)).squeeze(0)  # view(-1,1,128,128)
    x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=3)
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    mask = (x != 0)
    x = transform_image(x) * mask
    patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
    # print(x.shape)
    batch_size = 32
    res = []
    for i in range(0, x.shape[0], batch_size):
        batch = x[i:i + batch_size]  # [B, 3, H, W]
        if mode == 'dino':
            out = backbone(batch, output_hidden_states=True)  # sam.image_encoder(batch)
            feature_map = out.last_hidden_state[:, 5:, :]
            feature_map = feature_map.view(-1, patch_h, patch_w, feature_map.shape[-1]).permute(0, 3, 1, 2)  #
        else:
            feature_map = sam.image_encoder(batch)

        res.append(feature_map)
    patch_features = torch.cat(res, dim=0)

    # patch_features = patch_features.permute(0, 3, 1, 2)  # [batch, dim, H, W]

    patch_features = torch.permute(patch_features, (1, 2, 3, 0)).unsqueeze(0)

    patch_features = F.interpolate(patch_features, scale_factor=(8, 8, 1), mode='trilinear', align_corners=True)
    # feature_map = feature_map[:, :, pad_h//2: pad_h//2 + H//2, pad_w//2: pad_w//2 + W//2, :]
    patch_features = patch_features[:, :, pad_h: pad_h + H, pad_w: pad_w + W, :]
    # print(feature_map.shape)
    return patch_features


def torch_pca(x, n_components=256):
    """
    PCA implementation using PyTorch (avoids CPU-GPU transfer)
    """
    # Center the data
    x_centered = x - x.mean(dim=0)

    # SVD
    U, S, V = torch.svd_lowrank(x_centered, q=n_components)

    # Project to lower dimension
    x_reduced = torch.matmul(x_centered, V)

    return x_reduced, V


def apply_pca_to_3d_features_torch(x, y, n_components=256, model_path="pca_model.pkl", standardize=True):
    """
    Ultra-fast PCA using PyTorch (no CPU-GPU transfer)
    """
    if os.path.exists(model_path):
        models = torch.load(model_path)
        pca_components = models['pca_components']
        mean = models['mean']
        scaler_mean = models.get('scaler_mean', None)
        scaler_std = models.get('scaler_std', None)
        # print(f"Loaded PCA model from {model_path}")
        fit_new_models = False
    else:
        pca_components = None
        mean = None
        scaler_mean = None
        scaler_std = None
        fit_new_models = True
        # print("Fitting new PCA model")

    # Original shape
    w, h, d = x.shape[2:]
    device = x.device

    # Reshape
    x_flat = x.contiguous().view(768, -1).T  # (num_voxels, 768)
    y_flat = y.contiguous().view(768, -1).T  # (num_voxels, 768)
    '''
    # Standardization
    if standardize:
        if fit_new_models:
            scaler_mean = x_flat.mean(dim=0)
            scaler_std = x_flat.std(dim=0)
            scaler_std[scaler_std == 0] = 1.0  # avoid division by zero
            x_processed = (x_flat - scaler_mean) / scaler_std
        else:
            x_processed = (x_flat - scaler_mean) / scaler_std
        y_processed = (y_flat - scaler_mean) / scaler_std
    else:
        x_processed = x_flat
        y_processed = y_flat
    '''
    x_processed = x_flat
    y_processed = y_flat
    # PCA
    if fit_new_models:
        # Center data
        # mean = x_processed.mean(dim=0)
        # x_centered = x_processed - mean

        # SVD
        U, S, V = torch.svd_lowrank(x_processed, q=n_components)
        pca_components = V[:, :n_components]

        # Project
        x_reduced = torch.matmul(x_processed, pca_components)

        # Save model
        models = {
            'pca_components': pca_components,
            'mean': mean,
            'scaler_mean': scaler_mean,
            'scaler_std': scaler_std
        }
        torch.save(models, model_path)
        # print(f"Saved PCA model to {model_path}")
    else:
        # x_centered = x_processed - mean
        x_reduced = torch.matmul(x_processed, pca_components)

    # y_centered = y_processed - mean
    y_reduced = torch.matmul(y_processed, pca_components)

    # Reshape back
    x_reduced = x_reduced.T.contiguous().view(1, n_components, w, h, d)
    y_reduced = y_reduced.T.contiguous().view(1, n_components, w, h, d)

    # print(f"Input shapes: x{x.shape}, y{y.shape}")
    # print(f"Output shapes: x{x_reduced.shape}, y{y_reduced.shape}")

    return x_reduced, y_reduced


def coembed(x, y, flag='acdcreg', mode='sam'):
    if flag == 'acdcreg':
        emb_x = embed_acdc(x, mode)
        emb_y = embed_acdc(y, mode)
    else:
        emb_x = embed_abdomen(x, mode)
        emb_y = embed_abdomen(y, mode)

    if mode == 'dino':
        emb_x, emb_y = apply_pca_to_3d_features_torch(emb_x, emb_y, n_components=256)  # apply_pca_to_3d_features_torch
    x = torch.cat([x, emb_x], dim=1)
    y = torch.cat([y, emb_y], dim=1)
    return x, y


def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    # Getting model-related components
    test_loader = getters.getDataLoader(opt, split='test')
    model, _ = getters.getTestModelWithCheckpoints(opt)
    reg_model = registerSTModel(opt['img_size'], 'nearest').cuda()

    eval_dsc = AverageMeter()
    init_dsc = AverageMeter()
    eval_lv_dsc = AverageMeter()
    init_lv_dsc = AverageMeter()
    eval_rv_dsc = AverageMeter()
    init_rv_dsc = AverageMeter()
    eval_lvm_dsc = AverageMeter()
    init_lvm_dsc = AverageMeter()
    eval_jac_det = AverageMeter()
    eval_std_det = AverageMeter()
    eval_hd95 = AverageMeter()
    init_hd95 = AverageMeter()
    time_cal = 0

    if 1:
        upscale = [2, 2, 1]
        ss = opt['img_size']
        transformers = nn.ModuleList(
            [layers.SpatialTransformer((ss[0] // 2 ** i, ss[1] // 2 ** i, ss[2])).cuda() for i in range(5)])
        integrates = nn.ModuleList(
            [layers.VecInt((ss[0] // 2 ** i, ss[1] // 2 ** i, ss[2]), 7).cuda() for i in range(5)])

    df_data = []
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            # sub_idx = data[4].item()
            # sub_idx = data[6].item()
            # print(data[8])
            sub_idx = data[8]  # .item()
            data = [Variable(t.cuda()) for t in data[:6]]
            x, x_seg, x_seg_edt = data[0].float(), data[1].long(), data[4].float()
            y, y_seg, y_seg_edt = data[2].float(), data[3].long(), data[5].float()
            '''
            pos_flow1 = model(x,y,x,y,registration=True)
            pos_flow2 = model(y,x,y,x,registration=True)
            '''
            # 在代码开始前添加计时开始
            start_time = time.time()
            x_emb, y_emb = coembed(x, y)
            pos_flow1 = model(x_emb, y_emb, transformers, integrates, upscale, registration=True)
            pos_flow2 = model(y_emb, x_emb, transformers, integrates, upscale, registration=True)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"coembed and model execution time: {execution_time:.4f} seconds")
            time_cal = time_cal + execution_time
            print(f"----------ALL time: {time_cal:.4f} seconds")
            '''
            pos_flow1 = model(x_emb,y_emb, transformers, integrates, upscale,registration=True)
            pos_flow2 = model(y_emb,x_emb, transformers, integrates, upscale,registration=True)
            '''

            df_row = []

            def_out1 = reg_model(x_seg.cuda().float(), pos_flow1)
            dsc1, rv_dsc1, lvm_dsc1, lv_dsc1 = dice_eval(def_out1.long(), y_seg.long(), 4, output_individual=True)
            def_out2 = reg_model(y_seg.cuda().float(), pos_flow2)
            dsc2, rv_dsc2, lvm_dsc2, lv_dsc2 = dice_eval(def_out2.long(), x_seg.long(), 4, output_individual=True)
            dsc_p = (dsc1 + dsc2) / 2
            rv_dsc = (rv_dsc1 + rv_dsc2) / 2
            lvm_dsc = (lvm_dsc1 + lvm_dsc2) / 2
            lv_dsc = (lv_dsc1 + lv_dsc2) / 2
            eval_dsc.update(dsc_p.item(), x.size(0))
            eval_lv_dsc.update(lv_dsc.item(), x.size(0))
            eval_rv_dsc.update(rv_dsc.item(), x.size(0))
            eval_lvm_dsc.update(lvm_dsc.item(), x.size(0))

            dsc_i, rv_dsci, lvm_dsci, lv_dsci = dice_eval(x_seg.long(), y_seg.long(), 4, output_individual=True)
            init_dsc.update(dsc_i.item(), x.size(0))
            init_lv_dsc.update(lv_dsci.item(), x.size(0))
            init_rv_dsc.update(rv_dsci.item(), x.size(0))
            init_lvm_dsc.update(lvm_dsci.item(), x.size(0))

            df_row.append(sub_idx)
            df_row.append(dsc_p.item())
            df_row.append(dsc_i.item())
            df_row.append(rv_dsc.item())
            df_row.append(lvm_dsc.item())
            df_row.append(lv_dsc.item())

            # Jacobian determinant
            jac_det1 = jacobian_determinant(pos_flow1.detach().cpu().numpy())
            jac_det2 = jacobian_determinant(pos_flow2.detach().cpu().numpy())
            jac_det_val1 = computeJacDetVal(jac_det1, x_seg.shape[2:])
            jac_det_val2 = computeJacDetVal(jac_det2, x_seg.shape[2:])
            jac_det_val = (jac_det_val1 + jac_det_val2) / 2
            eval_jac_det.update(jac_det_val, x.size(0))

            # Standard deviation of log Jacobian determinant
            std_dev_jac1 = computeSDLogJ(jac_det1)
            std_dev_jac2 = computeSDLogJ(jac_det2)
            std_dev_jac = (std_dev_jac1 + std_dev_jac2) / 2
            eval_std_det.update(std_dev_jac, x.size(0))

            # Hausdorff distance 95
            moving = x_seg.long().squeeze().cpu().numpy()
            fixed = y_seg.long().squeeze().cpu().numpy()
            moving_warped = def_out1.long().squeeze().cpu().numpy()
            hd95_1 = compute_HD95(moving, fixed, moving_warped, 4, opt['voxel_spacing'])
            init_hd95_1 = compute_HD95(moving, fixed, moving, 4, opt['voxel_spacing'])

            moving = y_seg.long().squeeze().cpu().numpy()
            fixed = x_seg.long().squeeze().cpu().numpy()
            moving_warped = def_out2.long().squeeze().cpu().numpy()
            hd95_2 = compute_HD95(moving, fixed, moving_warped, 4, opt['voxel_spacing'])
            init_hd95_2 = compute_HD95(moving, fixed, moving, 4, opt['voxel_spacing'])

            hd95 = (hd95_1 + hd95_2) / 2
            eval_hd95.update(hd95, x.size(0))
            init_hd95_ = (init_hd95_1 + init_hd95_2) / 2
            init_hd95.update(init_hd95_, x.size(0))

            df_row.append(jac_det_val)
            df_row.append(std_dev_jac)
            df_row.append(hd95)
            df_row.append(init_hd95_)
            df_data.append(df_row)

            print(
                "Subject {} dice: {:.4f}, init dice: {:.4f}, rv dice: {:.4f}, lvm dice: {:.4f}, lv dice: {:.4f}, jac_det: {:.6f}, std_dev_jac: {:.4f}, hd95: {:.4f}, init hd95: {:.4f}".format(
                    sub_idx, dsc_p, dsc_i, rv_dsc, lvm_dsc, lv_dsc, jac_det_val, std_dev_jac, hd95, init_hd95_))

            if opt['is_save']:
                pos_flow1 = pos_flow1.permute(2, 3, 4, 0, 1).cpu().numpy()
                pos_flow2 = pos_flow2.permute(2, 3, 4, 0, 1).cpu().numpy()
                fp = os.path.join('logs', opt['dataset'], opt['model'], 'flow_fields')
                os.makedirs(fp, exist_ok=True)
                nib.save(nib.Nifti1Image(pos_flow1, None, None),
                         os.path.join(fp, '%s_flow_x2y.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(pos_flow2, None, None),
                         os.path.join(fp, '%s_flow_y2x.nii.gz' % (str(sub_idx).zfill(3))))
                '''
                warped_x2y = warped_x2y.squeeze().cpu().numpy()
                warped_y2x = warped_y2x.squeeze().cpu().numpy()
                fp = os.path.join('logs', opt['dataset'], opt['model'], 'warped_images')
                os.makedirs(fp, exist_ok=True)
                nib.save(nib.Nifti1Image(warped_x2y, None, None), os.path.join(fp,'%s_warped_x2y.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(warped_y2x, None, None), os.path.join(fp,'%s_warped_y2x.nii.gz' % (str(sub_idx).zfill(3))))
                '''
                x = x.squeeze().cpu().numpy()
                y = y.squeeze().cpu().numpy()
                fp = os.path.join('logs', opt['dataset'], opt['model'], 'images')
                os.makedirs(fp, exist_ok=True)
                nib.save(nib.Nifti1Image(x, None, None), os.path.join(fp, '%s_x.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(y, None, None), os.path.join(fp, '%s_y.nii.gz' % (str(sub_idx).zfill(3))))

    print("init dice: {:.7f}, init rv dice: {:.7f}, init lvm dice: {:.7f}, init lv dice: {:.7f}".format(init_dsc.avg,
                                                                                                        init_rv_dsc.avg,
                                                                                                        init_lvm_dsc.avg,
                                                                                                        init_lv_dsc.avg))

    print(
        "Average dice: {:.4f}, init dice: {:.4f}, rv dice: {:.4f}, lvm dice: {:.4f}, lv dice: {:.4f}, jac_det: {:.6f}, std_dev_jac: {:.4f}, hd95: {:.4f}, init hd95: {:.4f}".format(
            eval_dsc.avg, init_dsc.avg, eval_rv_dsc.avg, eval_lvm_dsc.avg, eval_lv_dsc.avg, eval_jac_det.avg,
            eval_std_det.avg, eval_hd95.avg, init_hd95.avg))

    df_row = ['Average', eval_dsc.avg, init_dsc.avg, eval_rv_dsc.avg, eval_lvm_dsc.avg, eval_lv_dsc.avg,
              eval_jac_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg]
    df_data.append(df_row)

    keys = ['subject', 'dice', 'init_dice', 'rv_dice', 'lvm_dice', 'lv_dice', 'jac_det', 'std_dev_jac', 'hd95',
            'init_hd95']
    df = pd.DataFrame(df_data, columns=keys)
    fp = os.path.join('logs', opt['dataset'], 'results_%s.csv' % opt['model'])
    df.to_csv(fp, index=False)


if __name__ == '__main__':
    opt = {
        'img_size': (128, 128, 16),  # input image size
        'in_shape': (128, 128, 16),  # input image size
        'logs_path': './logs',  # path to saved logs
        'num_workers': 4,  # number of workers for data loading
        'voxel_spacing': (1.8, 1.8, 10),  # voxel size
    }

    parser = argparse.ArgumentParser(description="cardiac")
    parser.add_argument("-m", "--model", type=str, default='voxelMorphComplex')
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-d", "--dataset", type=str, default='acdcreg')
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("-dp", "--datasets_path", type=str, default="/home/admin123/data/")
    parser.add_argument("--load_ckpt", type=str, default="best")  # best, last or epoch
    parser.add_argument("--is_save", type=int, default=0)  # whether to save the flow field
    parser.add_argument("--num_classes", type=int,
                        default=4)  # number of anatomical classes, 4 for cardiac, 14 for abdominal

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]: s.split('=')[1] for s in unknowns}

    run(opt)

'''
python test_cardiac.py -m dbgComplex -d acdcreg
python test_cardiac.py -d acdcreg --is_save 0 -m unetCardiacComplexDsc1S8 start_channel=8
python test_cardiacreg.py -d acdcreg -m RDP --is_save 0 --gpu_id 5
python test_cardiacreg.py -d acdcreg -m encoderOnlyACDCComplex --is_save 0 start_channel=32 --gpu_id 5 
python test_cardiacreg.py -d acdcreg -m memWarpComplex --is_save 0 start_channel=32 --gpu_id 5 
python test_cardiacreg.py -d acdcreg -m LessNet --is_save 0 start_channel=32 --gpu_id 5 
python test_cardiacreg.py -d acdcreg -m VxmLKUnetComplex --is_save 0 start_channel=32 --gpu_id 5 
python test_cardiacreg.py -d acdcreg -m voxelMorph --is_save 0 start_channel=32 --gpu_id 5 
python test_cardiacreg.py -d acdcreg -m SAMIR --is_save 0 start_channel=32 --gpu_id 5 
python test_cardiacreg_fbir.py -d acdcreg -m regdino_mlp --is_save 0 start_channel=32 --gpu_id 1
'''