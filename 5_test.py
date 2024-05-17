import os
import nibabel as nib
import skimage
from scipy.ndimage import zoom
from utils import *
import torch
import warnings
import scipy.ndimage as ndimage
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion,generate_binary_structure

# def norm_img(image, MIN_BOUND=0., MAX_BOUND=1000.):
#     image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
#     image[image > 1] = 1.
#     image[image < 0] = 0.
#     return image*255.

def window_transform(ct_array, windowWidth, windowCenter):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = newimg * 255.
    return newimg

if __name__ == '__main__':
    ############################
    # Parameters
    ############################

    IMAGEDIR = '/media/DataA/LiverVessel/test_dataset/public/ct/'
    LiverDIR = '/media/DataA/LiverVessel/test_dataset/public/origi_liver/'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    k = 11
    MODELPATH = 'checkpoints/MSENet1/net_1_best_epoch_98.pth'
    Architectures = ['MSENet','Cline','Baseline','Baseline_MS','MSENet']
    architecture = Architectures[-1]
    # 取块大小
    crop_x = 160
    crop_y = 160
    crop_z = 96
    S_xy = 80
    S_z = 48
    ############################
    # load data
    ###########################
    image_files = [os.path.join(IMAGEDIR, x)
                   for x in os.listdir(IMAGEDIR)
                   if x.endswith('.nii.gz')]

    Liver_files = [os.path.join(LiverDIR, x)
                   for x in os.listdir(LiverDIR)
                   if x.endswith('.nii.gz')]
    image_files.sort()
    Liver_files.sort()

    print(image_files)
    print(Liver_files)
    ############################
    # load the net
    ###########################
    from model.edge_attention_unet import EA_UNet
    from model.MSEnet import MSENet,Baseline,Baseline_MS
    # from model.unet import UNet

    def create_model(ema=False):
        model = MSENet(in_channels=1, out_channels=2)
        # model = model.to(device)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    print('#parameters:', sum(param.numel() for param in ema_model.parameters()))
    inet = ema_model.to(device)

    inet.load_state_dict(torch.load(MODELPATH))
    # state_dict = torch.load(MODELPATH)
    # new_state_dict = {}
    # for key, value in state_dict.items():
    #     new_key = key.replace('module.', '')
    #     new_state_dict[new_key] = value
    # inet.load_state_dict(state_dict)
    ############################
    # Test the net
    ############################
    inet.eval()
    with torch.no_grad():
        for idx in range(len(image_files)):
            image_name = image_files[idx].split('/')[-1].split('.')[0]
            print(image_name)
            image_itk = nib.load(image_files[idx])
            liver_itk = nib.load(Liver_files[idx])

            image_affine = image_itk.affine
            ctimage = image_itk.get_fdata()
            liver_affine = liver_itk.affine
            liver = liver_itk.get_fdata()

            img_size = ctimage.shape
            print(ctimage.shape,liver.shape)
            # struct = generate_binary_structure(3, 3)
            # ero_liver = binary_erosion(liver, structure=struct, iterations=1).astype('uint8')
            # seg_ero_liver = ero_liver == 1

            resampled = ndimage.zoom(ctimage, (-image_affine[0][0] / 1.0,
                                               -image_affine[1][1] / 1.0,
                                               image_affine[2][2] / 1.0), order=3)
            liver = ndimage.zoom(liver, (-liver_affine[0][0] / 1.0,
                                         -liver_affine[1][1] / 1.0,
                                          liver_affine[2][2] / 1.0), order=0)
            print(resampled.shape,liver.shape)

            # 图像归一化
            seg_liver = liver == 1
            ct_liver = resampled * seg_liver
            liver_min = ct_liver.min()
            liver_max = ct_liver.max()
            liver_wide = liver_max - liver_min
            liver_center = (liver_max + liver_min) / 2
            image = window_transform(resampled, liver_wide, liver_center)

            # 关注肝区
            indices = np.where(liver != 0)
            min_x, max_x, min_y, max_y, min_z, max_z = min(indices[0]), max(indices[0]), min(indices[1]), max(
                indices[1]), min(indices[2]), max(indices[2])
            # print(min_x, max_x, min_y, max_y, min_z, max_z)
            scan = image[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

            scan_sup = np.zeros([int(scan.shape[0] + (crop_x - scan.shape[0] % S_xy)),
                                 int(scan.shape[1] + (crop_y - scan.shape[1] % S_xy)),
                                 int(scan.shape[2] + (crop_z - scan.shape[2] % S_z))]).astype('int16')

            scan_sup[0:scan.shape[0], 0:scan.shape[1], 0:scan.shape[2]] = scan
            result_vessel_sup = np.zeros([int(scan.shape[0] + (crop_x - scan.shape[0] % S_xy)),
                                          int(scan.shape[1] + (crop_y - scan.shape[1] % S_xy)),
                                          int(scan.shape[2] + (crop_z - scan.shape[2] % S_z))]).astype('int16')

            print(image_name, scan.shape, scan_sup.shape)
            patch_count = 0


            for x in range(scan_sup.shape[0] // S_xy - 1):
                for y in range(scan_sup.shape[1] // S_xy - 1):
                    for z in range(scan_sup.shape[2] // S_z - 1):
                        patch = scan_sup[x * S_xy:x * S_xy + crop_x,
                                        y * S_xy:y * S_xy + crop_y,
                                        z * S_z:z * S_z + crop_z] / 255. # DHW
                        # print(patch.shape)
                        patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)  # BCDHW
                        # print(patch.shape)
                        pred1,cl,fm,_,_,_ = inet(patch)
                        # pred1,_ = inet(patch)
                        pred1 = torch.softmax(pred1, dim=1)
                        pred1 = torch.argmax(pred1, dim=1).cpu().data  # BCDHW
                        # print(type(pred1))
                        result_vessel_sup[x * S_xy:x * S_xy + crop_x, y * S_xy:y * S_xy + crop_y,z * S_z:z * S_z + crop_z] = pred1
                        print(patch_count, patch.shape, pred1.shape,np.unique(pred1))
                        patch_count = patch_count + 1

            result_vessel = result_vessel_sup[0:scan.shape[0], 0:scan.shape[1], 0:scan.shape[2]]
            result_vessel[result_vessel > 1] = 1
            result_vessel = measureimg(result_vessel, t_num=6)
            saved_image = np.zeros_like(resampled)
            strat_position = (min_x,min_y,min_z)
            saved_image[strat_position[0]:strat_position[0] + result_vessel.shape[0],
                        strat_position[1]:strat_position[1] + result_vessel.shape[1],
                        strat_position[2]:strat_position[2] + result_vessel.shape[2]]= result_vessel
            saved_image = saved_image * seg_liver
            # saved_image = skimage.transform.resize(saved_image, ctimage.shape, order=1, mode='reflect',  cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None)

            preprocess_scale_factor = (-image_affine[0][0] / 1.0,
                                       - image_affine[1][1] / 1.0,
                                       image_affine[2][2] / 1.0)
            restored_scale_factor = (1 / preprocess_scale_factor[0],
                                     1 / preprocess_scale_factor[1],
                                     1 / preprocess_scale_factor[2])
            saved_image = ndimage.zoom(saved_image, restored_scale_factor, order=0)
            #saved_image[saved_image > 0.5] =1
            
            if saved_image.shape != img_size:
                new_img = np.zeros_like(ctimage)
                new_img[0:saved_image.shape[0], 0:saved_image.shape[1], 0:saved_image.shape[2]] = saved_image
                savedImg1 = nib.Nifti1Image(new_img,image_affine)
            else:
                savedImg1 = nib.Nifti1Image(saved_image,image_affine)
            filepath = 'outputs/' + architecture + str(k)+'/'
            print(filepath)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            nib.save(savedImg1, filepath + image_name + '_vessel.nii.gz')



