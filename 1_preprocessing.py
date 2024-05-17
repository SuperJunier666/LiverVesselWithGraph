import os
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion,generate_binary_structure
import warnings

####################归一化###################
# def norm_img(image, MIN_BOUND=0., MAX_BOUND=1000.):
#     image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
#     image[image > 1] = 1.
#     image[image < 0] = 0.
#     return image * 255.

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

image_dir = '/media/Data/yanxc/Liver_vessel/dataset/public_dataset/CT'
label_dir = '/media/Data/yanxc/Liver_vessel/dataset/public_dataset/new_label2'
centerline_dir = '/media/Data/yanxc/Liver_vessel/dataset/public_dataset/cline2'
liver_dir = '/media/Data/yanxc/Liver_vessel/dataset/public_dataset/liver_mask'
##################读取文件名#######
image_paths = [os.path.join(image_dir, x)
                 for x in os.listdir(image_dir)
                 if x.endswith('.nii.gz')]

label_paths = [os.path.join(label_dir, x)
                    for x in os.listdir(label_dir)
                    if x.endswith('.nii.gz')]

centerline_paths = [os.path.join(centerline_dir, x)
                    for x in os.listdir(centerline_dir)
                    if x.endswith('.nii.gz')]
Liver_paths = [os.path.join(liver_dir, x)
                    for x in os.listdir(liver_dir)
                    if x.endswith('.nii.gz')]
image_paths.sort()
label_paths.sort()
centerline_paths.sort()
Liver_paths.sort()

print(len(image_paths),image_paths)
print(len(label_paths),label_paths)
print(len(centerline_paths),centerline_paths)
print(len(Liver_paths),Liver_paths)

###################取块参数###########################
crop_x = 160
crop_y = 160
crop_z = 96
S_xy = 40
S_z = 48
###################ITK ERROR改正##########################
##############第一次处理数据时运行########################
# for idx in range(len(centerline_paths)):
#     img = nib.load(centerline_paths[idx])
#     image_name = str(centerline_paths[idx].split('/')[-1].split('.')[0].split('_')[0])
#     qform = img.get_qform()
#     img.set_qform(qform)
#     sfrom = img.get_sform()
#     img.set_sform(sfrom)
#     nib.save(img, centerline_paths[idx])
#     print(image_name)
# print('No data errors')
###################################labeled_data###########################
for idx in range(len(image_paths)):
    # print(centerline_paths[idx].split('/')[-1].split('.')[0].split('_')[0])
    assert image_paths[idx].split('/')[-1].split('.')[0] == label_paths[idx].split('/')[-1].split('.')[0].split('_m')[0]
    assert  image_paths[idx].split('/')[-1].split('.')[0] == centerline_paths[idx].split('/')[-1].split('.')[0].split('_m')[0]
    assert image_paths[idx].split('/')[-1].split('.')[0] == Liver_paths[idx].split('/')[-1].split('.')[0].split('_l')[0]
    image_name = str(centerline_paths[idx].split('/')[-1].split('.')[0].split('_m')[0])

    image_itk = nib.load(image_paths[idx])
    label_itk = nib.load(label_paths[idx])
    centerline_itk = nib.load(centerline_paths[idx])
    Liver_itk = nib.load(Liver_paths[idx])

#     #图像矩阵
    image = image_itk.get_fdata()
    image_affine = image_itk.affine
    label = label_itk.get_fdata()
    centerline = centerline_itk.get_fdata()
    liver = Liver_itk.get_fdata()

    struct = generate_binary_structure(3, 3)
    ero_liver = binary_erosion(liver, structure=struct, iterations=1).astype('uint8')
    #图像归一化
    seg_liver = liver == 1
    ct_liver = image * liver
    liver_min = ct_liver.min()
    liver_max = ct_liver.max()
    liver_wide = liver_max - liver_min
    liver_center = (liver_max + liver_min) / 2
    image = window_transform(image,liver_wide,liver_center)

    #关注肝区
    indices = np.where(liver != 0)
    min_x, max_x, min_y, max_y, min_z, max_z = min(indices[0]), max(indices[0]), min(indices[1]), max(
        indices[1]), min(indices[2]), max(indices[2])
    label = label * ero_liver
    image = image[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]
    label = label[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]
    centerline = centerline[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

    # 图像扩增
    image_sup = np.zeros([int(image.shape[0] + (crop_x - image.shape[0] % S_xy)),
                          int(image.shape[1] + (crop_y - image.shape[1] % S_xy)),
                          int(image.shape[2] + (crop_z - image.shape[2] % S_z))]).astype('int16')
    # image_sup[:] = 170
    image_sup[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image

    label_sup = np.zeros([int(label.shape[0] + (crop_x - label.shape[0] % S_xy)),
                          int(label.shape[1] + (crop_y - label.shape[1] % S_xy)),
                          int(label.shape[2] + (crop_z - label.shape[2] % S_z))]).astype('int16')
    label_sup[0:label.shape[0], 0:label.shape[1], 0:label.shape[2]] = label

    centerline_sup = np.zeros([int(centerline.shape[0] + (crop_x - centerline.shape[0] % S_xy)),
                          int(centerline.shape[1] + (crop_y - centerline.shape[1] % S_xy)),
                          int(centerline.shape[2] + (crop_z - centerline.shape[2] % S_z))]).astype('int16')
    centerline_sup[0:centerline.shape[0], 0:centerline.shape[1], 0:centerline.shape[2]] = centerline

    print(image_name, image.shape, image_sup.shape, label_sup.shape,centerline_sup.shape)
#
    patch_count = 1
    save_path = '/media/Data/yanxc/Liver_vessel/pre_data/' + image_name + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
##########################滑窗取块##############################################
    for x in range(image_sup.shape[0] // S_xy - 1):
        for y in range(image_sup.shape[1] // S_xy - 1):
            for z in range(image_sup.shape[2] // S_z - 1):
                image_patch = image_sup[x * S_xy:x * S_xy + crop_x,
                                        y * S_xy:y * S_xy + crop_y,
                                        z * S_z:z * S_z + crop_z]  # DHW
                # print(np.unique(image_patch))
                label_patch = label_sup[x * S_xy:x * S_xy + crop_x,
                                        y * S_xy:y * S_xy + crop_y,
                                        z * S_z:z * S_z + crop_z]  # DHW
                centerline_patch = centerline_sup[x * S_xy:x * S_xy + crop_x,
                                                  y * S_xy:y * S_xy + crop_y,
                                                  z * S_z:z * S_z + crop_z]  # DHW
                if np.sum(label_patch)> 5000:
                    savedImg = nib.Nifti1Image(image_patch, image_affine)
                    savedSeg = nib.Nifti1Image(label_patch, image_affine)
                    savedCenL = nib.Nifti1Image(centerline_patch, image_affine)

                    nib.save(savedImg,save_path + image_name + '_patch' + str(patch_count).zfill(3) + '_img.nii.gz')
                    nib.save(savedSeg,save_path + image_name + '_patch' + str(patch_count).zfill(3) + '_seg.nii.gz')
                    nib.save(savedCenL,save_path + image_name + '_patch' + str(patch_count).zfill(3) + '_cenertline.nii.gz')
                    print('patch:', patch_count, image_patch.shape, label_patch.shape,centerline_patch.shape)
                    patch_count = patch_count + 1
    print(image_name, 'finished!\n')

