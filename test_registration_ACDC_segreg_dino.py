import os
import torch
import interpol
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
from torch.autograd import Variable

from utils import getters, setters
from utils.functions import AverageMeter, registerSTModel, dice_binary, dice_eval, \
    computeJacDetVal, computeSDLogJ, compute_HD95, jacobian_determinant
import torch.nn.functional as F



import matplotlib.pyplot as plt

def visualize_slice(img, mask=None, save_path='res.png', batch=0, channel=0, depth=8, overlay=True):
    slice_img = img[batch, channel, :, :, depth].cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(slice_img*255, cmap="gray")
    plt.title(f"Image Slice (B={batch}, C={channel}, D={depth})")
    plt.axis("off")

    if mask is not None:
        # 
        if mask.shape[1] > 1:  
            slice_mask = torch.argmax(mask[batch, :, :, :, depth], dim=0).cpu().numpy()
        else:
            slice_mask = mask[batch, channel, :, :, depth].cpu().numpy()

        if overlay:
            plt.imshow(slice_mask, cmap="jet", alpha=0.5)  # 
        else:
            plt.figure(figsize=(6,6))
            plt.imshow(slice_mask, cmap="jet")
            plt.title(f"Mask Slice (B={batch}, C={channel}, D={depth})")
            plt.axis("off")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    print('saved')
    return

###########dino
from sklearn.decomposition import PCA
from einops import repeat, rearrange
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
transform_image = transforms.Compose([
    #transforms.Resize(image_size),
    #transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#backbone = torch.hub.load('/data/data4/hy/dinov3-vitb16-pretrain-lvd1689m','dinov3_vitb16', source='local', weights='model.safetensors')
pretrained_model_name = "/data/data4/hy/dinov3-vitb16-pretrain-lvd1689m"

#processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
backbone = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
        }

encoder_size = 'base'
backbone = backbone
#self.head = DPTHead(2, self.backbone.embed_dim, 128, False, out_channels=[96, 192, 384, 768]) #model.config.hidden_size
#self.head = DPTHead(2, self.backbone.config.hidden_size, 128, False, out_channels=[96, 192, 384, 768]) #model.config.hidden_size
for p in backbone.parameters():
    p.requires_grad = False
###########
def embed(x):
    B,C,H,W,D = x.shape
    x = torch.permute(x,(0,4,1,2,3)).view(-1,C,H,W)
    x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=3)
    x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
    x = transform_image(x)
    patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
    #print(x.shape)
    outputs = backbone(x, output_hidden_states=True)
    intermediate_features = outputs.hidden_states 
    patch_features = outputs.last_hidden_state[:, 5:, :]
    #pooled_output = outputs.hidden_states[2]
    #print(patch_features.shape, pooled_output.shape,pooled_output[0,0,2:3],pooled_output[0,2:5,2:3],pooled_output[0,5:7,2:3],pooled_output[0,-1,2:3],pooled_output[0,-2,2:3])
    feature_map = patch_features.reshape(-1, patch_h, patch_w, patch_features.shape[-1])
    feature_map = feature_map.permute(0, 3, 1, 2)  # [batch, dim, H, W]
    #print(len(intermediate_features),intermediate_features[2].shape,intermediate_features[5].shape,intermediate_features[8].shape,intermediate_features[11].shape)
    feature_map = torch.permute(feature_map,(1,2,3,0))#.unsqueeze(0)
    return feature_map
n_components = 4
pca = PCA(n_components=n_components, whiten=True)
def dino_coseg(x,y):
    x_emb = embed(x)
    y_emb = embed(y)
    dim, w, h, d = x_emb.shape
    #print(x_emb.shape)
    x_emb = x_emb.contiguous().view(dim, -1).permute(1, 0).cpu().numpy()
    y_emb = y_emb.contiguous().view(dim, -1).permute(1, 0).cpu().numpy()
    pca.fit(y_emb)
    x_emb = pca.transform(x_emb)
    x_emb = torch.from_numpy(x_emb).view(w, h, d, n_components).cuda()
    x_emb = torch.sigmoid(x_emb * 2.0).permute(3, 0, 1, 2)  # [3, H, W]
    y_emb = pca.transform(y_emb)
    y_emb = torch.from_numpy(y_emb).view(w, h, d, n_components).cuda()
    y_emb = torch.sigmoid(y_emb * 2.0).permute(3, 0, 1, 2)  # [3, H, W]
    #print(x_emb.shape)
    x_emb = F.interpolate(x_emb.unsqueeze(0), scale_factor=(4, 4, 1), mode='trilinear', align_corners=True)
    y_emb = F.interpolate(y_emb.unsqueeze(0), scale_factor=(4, 4, 1), mode='trilinear', align_corners=True)
    x_emb= torch.argmax(x_emb, dim=1).unsqueeze(0)
    y_emb= torch.argmax(y_emb, dim=1).unsqueeze(0)
    return x_emb, y_emb
    


def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    # Getting model-related components
    test_loader = getters.getDataLoader(opt, split='test')
    model, _ = getters.getTestModelWithCheckpoints(opt)
    reg_model = registerSTModel(opt['img_size'], 'nearest').cuda()
    reg_model_img = registerSTModel(opt['img_size'], 'bilinear').cuda()

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

    df_data = []
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            sub_idx = data[8].item()
            data = [Variable(t.cuda()) for t in data[:8]]
            x, x_seg = data[0].float(), data[1]
            y, y_seg = data[2].float(), data[3]
            x_edt, y_edt = data[4].float(), data[5]
            x_autoseg, y_autoseg = data[6].float(), data[7]
            '''
            pos_flow1 = model(x,y,x,y,registration=True)
            pos_flow2 = model(y,x,y,x,registration=True)
            '''
            pos_flow1 = torch.zeros(1, 3, opt['in_shape'][0], opt['in_shape'][1], opt['in_shape'][2]).cuda()
            pos_flow2 = torch.zeros(1, 3, opt['in_shape'][0], opt['in_shape'][1], opt['in_shape'][2]).cuda()
            
            #tmp_x_seg, tmp_y_seg = x_seg.clone(), y_seg.clone()
            tmp_x_seg, tmp_y_seg = dino_coseg(x,y)
            mode = 3
            if mode == 2:
                tmp_x_seg[tmp_x_seg>0] = 1
                tmp_y_seg[tmp_y_seg>0] = 1
            elif mode ==3:
                tmp_x_seg[tmp_x_seg==3] = 2
                tmp_y_seg[tmp_y_seg==3] = 2
            visualize_slice(x, tmp_x_seg, 'xseg.png')
            visualize_slice(y, tmp_y_seg, 'yseg.png')
            input()
            for i in range(4):
                pos_flow_tmp1 = model(x * (tmp_x_seg == i), y * (tmp_y_seg == i), (tmp_x_seg == i), (tmp_y_seg == i), x_edt * (tmp_x_seg == i), y_edt * (tmp_y_seg == i), registration=True)
                #pos_flow_tmp1 = model(x * (tmp_x_seg == i), y * (tmp_y_seg == i), (tmp_x_seg == i), (tmp_y_seg == i), registration=True)
                pos_flow1 = pos_flow1 * (tmp_y_seg != i) + pos_flow_tmp1 * (tmp_y_seg == i)

                pos_flow_tmp2 = model(y * (tmp_y_seg == i), x * (tmp_x_seg == i), (tmp_y_seg == i), (tmp_x_seg == i), y_edt * (tmp_y_seg == i), x_edt * (tmp_x_seg == i), registration=True)
                #pos_flow_tmp2 = model(y * (tmp_y_seg == i), x * (tmp_x_seg == i), (tmp_y_seg == i), (tmp_x_seg == i), registration=True)
                pos_flow2 = pos_flow2 * (tmp_x_seg != i) + pos_flow_tmp2 * (tmp_x_seg == i)

            # pos_flow1 = model(x, y, x_seg, y_seg, registration=True)
            # pos_flow2 = model(y, x, y_seg, x_seg, registration=True)

            df_row = []

            def_out1 = reg_model(x_seg.cuda().float(), pos_flow1)
            warped_x2y = reg_model_img(x.cuda().float(), pos_flow1)
            dsc1, rv_dsc1, lvm_dsc1, lv_dsc1 = dice_eval(def_out1.long(), y_seg.long(), 4, output_individual=True)
            def_out2 = reg_model(y_seg.cuda().float(), pos_flow2)
            warped_y2x = reg_model_img(y.cuda().float(), pos_flow2)
            dsc2, rv_dsc2, lvm_dsc2, lv_dsc2 = dice_eval(def_out2.long(), x_seg.long(), 4, output_individual=True)
            print(dsc1,dsc2)
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

            print("Subject {} dice: {:.4f}, init dice: {:.4f}, rv dice: {:.4f}, lvm dice: {:.4f}, lv dice: {:.4f}, "
                  "jac_det: {:.6f}, std_dev_jac: {:.4f}, hd95: {:.4f}, init hd95: {:.4f}".
                  format(sub_idx, dsc_p, dsc_i, rv_dsc, lvm_dsc, lv_dsc, jac_det_val, std_dev_jac, hd95, init_hd95_))

            if opt['is_save']:
                pos_flow1 = pos_flow1.permute(2, 3, 4, 0, 1).cpu().numpy()
                pos_flow2 = pos_flow2.permute(2, 3, 4, 0, 1).cpu().numpy()
                fp = os.path.join('logs', opt['dataset'], opt['model'], 'flow_fields')
                os.makedirs(fp, exist_ok=True)
                nib.save(nib.Nifti1Image(pos_flow1, None, None), os.path.join(fp, '%s_flow_x2y.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(pos_flow2, None, None), os.path.join(fp, '%s_flow_y2x.nii.gz' % (str(sub_idx).zfill(3))))
                warped_x2y = warped_x2y.squeeze().cpu().numpy()
                warped_y2x = warped_y2x.squeeze().cpu().numpy()
                warped_seg_x = def_out1.squeeze().cpu().numpy()
                warped_seg_y = def_out2.squeeze().cpu().numpy()
                fp = os.path.join('logs', opt['dataset'], opt['model'], 'warped_images')
                os.makedirs(fp, exist_ok=True)
                nib.save(nib.Nifti1Image(warped_x2y, None, None), os.path.join(fp, '%s_warped_x2y.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(warped_y2x, None, None), os.path.join(fp, '%s_warped_y2x.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(warped_seg_x, None, None), os.path.join(fp, '%s_warpedseg_x2y.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(warped_seg_y, None, None), os.path.join(fp, '%s_warpedseg_y2x.nii.gz' % (str(sub_idx).zfill(3))))
                x = x.squeeze().cpu().numpy()
                y = y.squeeze().cpu().numpy()
                fp = os.path.join('logs', opt['dataset'], opt['model'], 'images')
                os.makedirs(fp, exist_ok=True)
                nib.save(nib.Nifti1Image(x, None, None), os.path.join(fp, '%s_x.nii.gz' % (str(sub_idx).zfill(3))))
                nib.save(nib.Nifti1Image(y, None, None), os.path.join(fp, '%s_y.nii.gz' % (str(sub_idx).zfill(3))))

    print("init dice: {:.7f}, init rv dice: {:.7f}, init lvm dice: {:.7f}, init lv dice: {:.7f}"
          .format(init_dsc.avg, init_rv_dsc.avg, init_lvm_dsc.avg, init_lv_dsc.avg))

    print("Average dice: {:.4f}, init dice: {:.4f}, rv dice: {:.4f}, lvm dice: {:.4f}, lv dice: {:.4f},"
          " jac_det: {:.6f}, std_dev_jac: {:.4f}, hd95: {:.4f}, init hd95: {:.4f}"
          .format(eval_dsc.avg, init_dsc.avg, eval_rv_dsc.avg, eval_lvm_dsc.avg,
                  eval_lv_dsc.avg, eval_jac_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg))

    df_row = ['Average', eval_dsc.avg, init_dsc.avg, eval_rv_dsc.avg, eval_lvm_dsc.avg, eval_lv_dsc.avg,
              eval_jac_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg]
    df_data.append(df_row)

    keys = ['subject', 'dice', 'init_dice', 'rv_dice', 'lvm_dice', 'lv_dice', 'jac_det',
            'std_dev_jac', 'hd95', 'init_hd95']
    df = pd.DataFrame(df_data, columns=keys)
    fp = os.path.join('logs', opt['dataset'], 'results_%s.csv' % opt['model'])
    df.to_csv(fp, index=False)

if __name__ == '__main__':

    opt = {
        'img_size': (128, 128, 16),  # input image size
        'in_shape': (128, 128, 16),  # input image size
        'logs_path': './logs',     # path to saved logs
        'num_workers': 4,          # number of workers for data loading
        'voxel_spacing': (1.8, 1.8, 10)   # voxel size
    }

    parser = argparse.ArgumentParser(description="cardiac")
    parser.add_argument("-m", "--model", type=str, default='SegReg')
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-d", "--dataset", type=str, default='acdcreg')
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("-dp", "--datasets_path", type=str, default="./../../../data/")
    parser.add_argument("--load_ckpt", type=str, default="best")  # best, last or epoch
    parser.add_argument("--is_save", type=int, default=0)  # whether to save the flow field

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
python test_registration_ACDC_segreg_num.py -d acdcreg -m voxelMorph --is_save 0 --gpu_id 5 
'''