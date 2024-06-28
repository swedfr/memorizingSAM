
import os
join = os.path.join
import numpy as np
from glob import glob
import torch
from segment_anything.modeling.imageencoderm3D import ImageEncoderViT3D
from tqdm import tqdm
import argparse
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SimpleITK as sitk
import torchio as tio
import numpy as np
from collections import OrderedDict, defaultdict
import json
import pickle
from functools import partial
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL_Val

parser = argparse.ArgumentParser()
parser.add_argument('-tdp', '--test_data_path', type=str, default='/content/drive/MyDrive/lighting_sam_3d/data/process_train_data/word')
parser.add_argument('-vp', '--vis_path', type=str, default='/content/drive/MyDrive/paper_visual_results/totalseg0441/med_sam_2d')
parser.add_argument('-cp', '--checkpoint_path', type=str, default='/content/drive/MyDrive/lighting_sam_3d/vertebraall/sam_model_latest.pth')
parser.add_argument('-count','--number',type = int,default = 0)

parser.add_argument('-sn', '--save_name', type=str, default='/content/drive/MyDrive/paper_visual_results/totalseg2d.py')

parser.add_argument('--image_size', type=int, default=256)  #
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
parser.add_argument('-nc', '--num_clicks', type=int, default=10)
parser.add_argument('-pm', '--point_method', type=str, default='default')
parser.add_argument('-dt', '--data_type', type=str, default='Tr')
parser.add_argument("--encoder_adapter", type=bool, default=False, help="use adapter")
parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--ft2d', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=2023)

args = parser.parse_args()

SEED = args.seed
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.init()

click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
}

if __name__ == "__main__":   
    all_dataset_paths = glob(join(args.test_data_path))
    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")
    


    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.crop_size,args.crop_size,args.crop_size)),
    ]
    
    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths, 
        mode="Val", 
        data_type=args.data_type, 
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )

    checkpoint_path = args.checkpoint_path

    device = args.device
    print("device:", device)
    vitmodel = ImageEncoderViT3D(                                    #
            depth=6,
            embed_dim=768,
            img_size=128,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=6,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[],
            window_size=0,
            out_chans=384,
            skip_layer = 2,
        ).to(device)
    model_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = model_dict['model_state_dict']
    vitmodel.load_state_dict(state_dict,strict = False)

    datapath = '/content/drive/MyDrive/lighting_sam_3d/ckpt/FastSAM3DMemory'
    k = args.number
    for batch_data in tqdm(test_dataloader):
        os.makedirs(os.path.join(datapath,str(k)),exist_ok = True)

        image3D, gt3D, img_name = batch_data
        image3D = image3D.float()
        image3D=image3D.to(device)
        sz = image3D.size()
        torch.save(image3D,os.path.join(datapath,str(k),"image.pt"))
        if(sz[2]<args.crop_size or sz[3]<args.crop_size or sz[4]<args.crop_size):
            print("[ERROR] wrong size", sz, "for", img_name)
        image_embedding = vitmodel(image3D,os.path.join(datapath,str(k)))         
        k += 1   