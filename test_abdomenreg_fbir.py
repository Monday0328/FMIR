import os
import torch
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from torch.autograd import Variable
import torch.nn.functional as F

from utils import getters, setters
from utils.mappers import label2text_dict_abdomenct as label2text_dict
from utils.functions import AverageMeter, registerSTModel, dice_eval, dice_binary, jacobian_determinant, compute_HD95
from monai.inferers import SlidingWindowInferer
from scipy.ndimage import zoom
import nibabel as nib


import torch
torch.cuda.set_device(0)  # 强制使用GPU 5



import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(8) 
torch.set_default_dtype(torch.float32)


#########SAM
from models.segment_anything import build_sam, SamPredictor
from models.segment_anything import sam_model_registry
from models.segment_anything.modeling import Sam
from einops import repeat, rearrange
#SAM config
#interpolate = BilinearInterpolation(scale_factor=(0.5, 0.5, 1))
vit_name = 'vit_h'
# sam_ckpt = '/data/data1/hy/code/MA-SAM-main/sam_vit_h_4b8939.pth'
sam_ckpt = '/data/data1/hy/code/MA-SAM-main/sam_vit_h_4b8939.pth'
num_classes = 4
sam_size = 512 #self.inshape
        
sam = sam_model_registry[vit_name](image_size=sam_size,
                                                num_classes=num_classes,
                                                checkpoint=sam_ckpt, in_channel=3,
                                                pixel_mean=[0.5, 0.5, 0.5], pixel_std=[0.5, 0.5, 0.5]).cuda()
for p in sam.parameters():
    p.requires_grad = False
#########

###########dino
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.voxelmorph.torch import layers
from transformers import AutoImageProcessor, AutoModel, Dinov2Config, AutoConfig
from einops import repeat, rearrange
from torchvision import transforms
transform_image = transforms.Compose([
    #transforms.Resize(image_size),
    #transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#backbone = torch.hub.load('/data/data4/hy/dinov3-vitb16-pretrain-lvd1689m','dinov3_vitb16', source='local', weights='model.safetensors')
pretrained_model_name = "/data/data4/hy/dinov3-vitb16-pretrain-lvd1689m"
'''
config = Dinov3Config.from_pretrained(pretrained_model_name)
config.image_size = 256  # 
config.patch_size = 16   #
'''
config = AutoConfig.from_pretrained(pretrained_model_name)
#print(config)
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

def embed_acdc(x):
    B,C,H,W,D = x.shape
    x = torch.permute(x,(0,4,1,2,3)).view(-1,C,H,W)
    x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=3)
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
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
    feature_map = torch.permute(feature_map,(1,2,3,0)).unsqueeze(0)
    feature_map = F.interpolate(feature_map, scale_factor=(8, 8, 1), mode='trilinear', align_corners=True)
    return feature_map


def embed_abdomen(x,mode):
    B,C,H,W,D = x.shape
    target_size = 128  
    pad_h = (target_size - H) // 2
    pad_h_after = (target_size - H) // 2
    pad_w = (target_size - W) // 2
    pad_w_after = (target_size - W) // 2
    
    x = F.pad(x,
        #pad=(0, 0, pad_w, pad_w_after, pad_h, pad_h_after),  # W, H, D
        pad=(0,0, pad_w, pad_w_after,pad_h, pad_h_after),
        mode='constant',
        value=0).cuda()
    #print(x.shape)#96, 80, 128
    x = torch.permute(x,(0,4,1,2,3)).squeeze(0)#view(-1,1,128,128)
    x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=3)
    x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
    mask = (x!=0)
    x = transform_image(x)*mask
    patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
    #print(x.shape)
    batch_size = 32
    res = []
    for i in range(0, x.shape[0], batch_size): 
        batch = x[i:i+batch_size]  # [B, 3, H, W]
        if mode == 'dino':
            out = backbone(batch, output_hidden_states=True) #sam.image_encoder(batch)
            feature_map = out.last_hidden_state[:, 5:, :]
            feature_map = feature_map.view(-1, patch_h, patch_w, feature_map.shape[-1]).permute(0, 3, 1, 2)  # 
        else:
            feature_map = sam.image_encoder(batch)
        
        res.append(feature_map)
    patch_features = torch.cat(res, dim=0)
    
    #patch_features = patch_features.permute(0, 3, 1, 2)  # [batch, dim, H, W]
    
    patch_features = torch.permute(patch_features,(1,2,3,0)).unsqueeze(0)
    
    patch_features = F.interpolate(patch_features,scale_factor=(4, 4,1),  mode='trilinear',align_corners=True)
    #feature_map = feature_map[:, :, pad_h//2: pad_h//2 + H//2, pad_w//2: pad_w//2 + W//2, :]
    patch_features = patch_features[:, :, pad_h: pad_h + H, pad_w: pad_w + W, :]
    #print(feature_map.shape)
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
        #print(f"Loaded PCA model from {model_path}")
        fit_new_models = False
    else:
        pca_components = None
        mean = None
        scaler_mean = None
        scaler_std = None
        fit_new_models = True
        #print("Fitting new PCA model")
    
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
        #mean = x_processed.mean(dim=0)
        #x_centered = x_processed - mean
        
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
        #print(f"Saved PCA model to {model_path}")
    else:
        #x_centered = x_processed - mean
        x_reduced = torch.matmul(x_processed, pca_components)
    
    #y_centered = y_processed - mean
    y_reduced = torch.matmul(y_processed, pca_components)
    
    # Reshape back
    x_reduced = x_reduced.T.contiguous().view(1, n_components, w, h, d)
    y_reduced = y_reduced.T.contiguous().view(1, n_components, w, h, d)
    
    #print(f"Input shapes: x{x.shape}, y{y.shape}")
    #print(f"Output shapes: x{x_reduced.shape}, y{y_reduced.shape}")
    
    return x_reduced, y_reduced

def coembed(x, y,flag='acdcreg',mode='dino', train = False):
    print(mode, "~~~~~~~~~~~~~~mode")
    if flag == 'acdcreg':
        emb_x = embed_acdc(x,mode)
        emb_y = embed_acdc(y,mode)
    else:
        emb_x = embed_abdomen(x,mode)
        emb_y = embed_abdomen(y,mode)
    
    if mode == 'dino':
        if train == True:
            indices = random.sample(range(0, 768 + 1), 256)
            emb_x = emb_x.index_select(1, indices)
            emb_y = emb_y.index_select(1, indices)
        else:
            emb_x, emb_y = apply_pca_to_3d_features_torch(emb_x, emb_y, n_components=256) #apply_pca_to_3d_features_torch
    x = torch.cat([x,emb_x], dim=1)
    y = torch.cat([y,emb_y], dim=1)
    return x,y

def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    model, _ = getters.getTestModelWithCheckpoints(opt)
    # 然后确保模型和数据都在这个设备上
    model = model.cuda()  # 这会自动使用当前设置的设备

    test_loader = getters.getDataLoader(opt, split=opt['field_split'])
    reg_model_ne = registerSTModel(opt['img_size'], 'nearest').cuda()
    reg_model_ti = registerSTModel(opt['img_size'], 'bilinear').cuda()

    organ_eval_dsc = [AverageMeter() for i in range(1,14)]
    eval_dsc = AverageMeter()
    init_dsc = AverageMeter()
    eval_det = AverageMeter()
    eval_std_det = AverageMeter()
    eval_hd95 = AverageMeter()
    init_hd95 = AverageMeter()
    df_data = []
    if 1:
        upscale = [2,2,2]
        ss = opt['img_size']
        print(ss)
        transformers = nn.ModuleList([layers.SpatialTransformer((ss[0]//2**i,ss[1]//2**i,ss[2]//2**i)).cuda() for i in range(5)])
        integrates = nn.ModuleList([layers.VecInt((ss[0]//2**i,ss[1]//2**i,ss[2]//2**i), 7).cuda() for i in range(5)])
            
    
    
    
    keys = ['idx1', 'idx2'] + [label2text_dict[i] for i in range(1,14)] + ['val_dice', 'init_dice', 'jac_det', 'std_dev', 'hd95', 'init_hd95']
    pid = 0
    with torch.no_grad():
        for num, data in enumerate(test_loader):
            model.eval()
            idx1, idx2 = data[4][0].item(), data[5][0].item()
            loop_df_data = [idx1, idx2]
            data = [Variable(t.cuda()) for t in data[:4]]
            x, x_seg = data[0].float(), data[1].long()
            y, y_seg = data[2].float(), data[3].long()
            '''
            data = [Variable(t.cuda()) for t in data[:10]]
            x, x_seg, x_seg_edt, x_seg_auto = data[0].float(), data[1].long(), data[6].float(), data[8].long()
            y, y_seg, y_seg_edt, y_seg_auto = data[2].float(), data[3].long(), data[7].float(), data[9].long()
            '''
            x_emb,y_emb = coembed(x,y,opt['dataset'])

            #pos_flow = model(x,y, transformers, integrates, upscale,registration=True)  
            pos_flow = model(x_emb,y_emb, transformers, integrates, upscale,registration=True)      
            f_xy = pos_flow
            def_y = reg_model_ti(x, f_xy)
            def_out = reg_model_ne(x_seg.float(), f_xy)
            for idx in range(1,14):
                dsc_idx = dice_binary(def_out.long().squeeze().cpu().numpy(), y_seg.long().squeeze().cpu().numpy(), idx)
                loop_df_data.append(dsc_idx)
                organ_eval_dsc[idx-1].update(dsc_idx, x.size(0))
            dsc1 = dice_eval(def_out.long(), y_seg.long(), 14)
            eval_dsc.update(dsc1.item(), x.size(0))
            dsc2 = dice_eval(x_seg.long(), y_seg.long(), 14)
            #dsc2 = dice_eval(y_seg_auto.long(), y_seg.long(), 14)
            init_dsc.update(dsc2.item(), x.size(0))

            jac_det = jacobian_determinant(f_xy.detach().cpu().numpy())
            jac_det_val = np.sum(jac_det <= 0) / np.prod(x_seg.shape)
            eval_det.update(jac_det_val, x.size(0))

            log_jac_det = np.log(np.abs((jac_det+3).clip(1e-8, 1e8)))
            std_dev_jac = np.std(log_jac_det)
            eval_std_det.update(std_dev_jac, x.size(0))

            moving = x_seg.long().squeeze().cpu().numpy()
            fixed = y_seg.long().squeeze().cpu().numpy()
            moving_warped = def_out.long().squeeze().cpu().numpy()
            hd95_1 = compute_HD95(moving, fixed, moving_warped,14,np.ones(3)*4)
            eval_hd95.update(hd95_1, x.size(0))
            hd95_2 = compute_HD95(moving, fixed, moving,14,np.ones(3)*4)
            init_hd95.update(hd95_2, x.size(0))

            print('idx1 {:d}, idx1 {:d}, val dice {:.4f}, init dice {:.4f}, jac det {:.4f}, std dev {:.4f}, hd95 {:.4f}, init hd95 {:.4f}'.format(idx1, idx2, dsc1.item(), dsc2.item(), jac_det_val, std_dev_jac, hd95_1, hd95_2))
            loop_df_data += [dsc1.item(), dsc2.item(), jac_det_val, std_dev_jac, hd95_1, hd95_2]
            df_data.append(loop_df_data)
            
            pid = pid + 1
            
            if num==35:
                print(num,'save')
                sv_x = x[0,0,:,:,:].cpu().numpy()
                sv_y = y[0,0,:,:,:].cpu().numpy()
                sv_xseg = x_seg[0,0,:,:,:].float().cpu().numpy()
                sv_yseg = y_seg[0,0,:,:,:].float().cpu().numpy()
                warp_x = def_y[0,0,:,:,:].cpu().numpy()
                warpd_xseg = def_out[0,0,:,:,:].long().float().cpu().numpy()
                flow_x2y = f_xy #F.interpolate(f_xy, scale_factor=2., mode='trilinear', align_corners=True)
                flow_x2y = flow_x2y.permute(2,3,4,0,1).cpu().numpy()

                fp = os.path.join(opt['log'],'%s_%s_reg' % (str(idx1).zfill(4), str(idx2).zfill(4)))
                os.makedirs(fp, exist_ok = True)

                nib.save(nib.Nifti1Image(sv_x, None, None), os.path.join(fp,'img_moving.nii.gz'))
                nib.save(nib.Nifti1Image(sv_y, None, None), os.path.join(fp,'img_fixed.nii.gz'))
                nib.save(nib.Nifti1Image(sv_xseg, None, None), os.path.join(fp,'seg_moving.nii.gz'))
                nib.save(nib.Nifti1Image(sv_yseg, None, None), os.path.join(fp,'seg_fixed.nii.gz'))
                nib.save(nib.Nifti1Image(warp_x, None, None), os.path.join(fp,'img_moving_warp.nii.gz'))
                nib.save(nib.Nifti1Image(warpd_xseg, None, None), os.path.join(fp,'seg_moving_warped.nii.gz'))
                nib.save(nib.Nifti1Image(flow_x2y, None, None), os.path.join(fp,'flow_x2y.nii.gz'))
            #

    avg_organ_eval_dsc = [organ_eval_dsc[i].avg for i in range(13)]
    avg_df_data = [0,0] + avg_organ_eval_dsc + [eval_dsc.avg, init_dsc.avg, eval_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg]
    df_data.append(avg_df_data)

    print('Avg val dice {:.4f}, Avg init dice {:.4f}, Avg jac det {:.4f}, Avg std dev {:.4f}, Avg hd95 {:.4f}, Avg init hd95 {:.4f}'.format(eval_dsc.avg, init_dsc.avg, eval_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg))

    df = pd.DataFrame(df_data, columns=keys)
    fp = os.path.join('logs', opt['dataset'], 'results_%s.csv' % opt['model'])
    df.to_csv(fp, index=False)

if __name__ == '__main__':

    opt = {
        'logs_path': './logs',
        'save_freq': 5,
        'n_checkpoints': 2,
        'num_workers': 4,
        #'img_size': (192//2,160//2,256//2),
        'in_shape': (192//2,160//2,256//2),
    }
    
    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("-m", "--model", type = str, default = 'VxmDense')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'abdomenreg')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "/data/data2/cx/data")
    parser.add_argument("--load_ckpt", type = str, default = "best") # best, last or epoch
    parser.add_argument("--field_split", type = str, default = 'test')
    parser.add_argument("--img_size", type = str, default = '(192//2,160//2,256//2)')
    parser.add_argument("--fea_type", type = str, default = 'unet')
    parser.add_argument("--num_classes", type=int, default=14)  #  number of anatomical classes, 4 for cardiac, 14 for abdominal

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    #opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    # 安全的参数解析
    opt['nkwargs'] = {}
    for s in unknowns:
        if '=' in s:
            parts = s.split('=')
            key = parts[0]
            value = '='.join(parts[1:])  # 处理包含多个等号的情况
            opt['nkwargs'][key] = value
        else:
            print(f"Warning: Ignoring invalid argument '{s}'. Use 'key=value' format.")

    #opt['nkwargs']['img_size'] = opt['img_size']
    opt['img_size'] = eval(opt['img_size'])
    

    run(opt)

'''
CUDA_VISIBLE_DEVICES=5 python test_abdomenreg.py -m encoderOnlyComplexS32 -d abdomenreg -bs 1 start_channel=32 --gpu_id 5
python test_abdomenreg.py -m encoderOnlyComplexS64 -d abdomenreg -bs 1 start_channel=64 --gpu_id 5
python test_abdomenreg.py -m encoderOnlyComplexS32R0.01 -d abdomenreg -bs 1 start_channel=32 --gpu_id 5
CUDA_VISIBLE_DEVICES=5 python test_abdomenreg.py -m RDP -d abdomenreg -bs 1 start_channel=32 --gpu_id 5
CUDA_VISIBLE_DEVICES=5 python test_abdomenreg.py -m VxmLKUnetComplex -d abdomenreg -bs 1 start_channel=32 --gpu_id 5
python test_abdomenreg.py -m LessNet -d abdomenreg -bs 1 start_channel=32 --gpu_id 5
python test_abdomenreg.py -m memWarpComplex -d abdomenreg -bs 1 start_channel=32 --gpu_id 5
python test_abdomenreg_fbir.py -m regdino_mlp -d abdomenreg -bs 1 start_channel=32 --gpu_id 5
'''
