import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
from skimage.transform import resize
import nibabel as nib
import warnings


def saved_preprocessed(savedImg,origin,direction,xyz_thickness,saved_path):
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
    sitk.WriteImage(newImg, saved_path)


image_dir = '/media/DataA/LiverVessel/dataset/public_dataset/ct/'
Liver_dir = '/media/DataA/LiverVessel/dataset/public_dataset/liver_mask/'
label_dir = '/media/DataA/LiverVessel/dataset/public_dataset/new_label/'
centerline_dir = '/media/DataA/LiverVessel/dataset/public_dataset/cline/'

image_paths = [os.path.join(image_dir, x)
                 for x in os.listdir(image_dir)
                 if x.endswith('.nii.gz')]
Liver_paths = [os.path.join(Liver_dir, x)
                    for x in os.listdir(Liver_dir)
                    if x.endswith('.nii.gz')]
label_paths = [os.path.join(label_dir, x)
                    for x in os.listdir(label_dir)
                    if x.endswith('.nii.gz')]
centerline_paths = [os.path.join(centerline_dir, x)
                    for x in os.listdir(centerline_dir)
                    if x.endswith('.nii.gz')]

image_paths.sort()
label_paths.sort()
Liver_paths.sort()
centerline_paths.sort()
config = {
        'xyz_thickness': [1.0, 1.0, 1.0],
        'expand_slice': 10,
    }

for idx in range(len(image_paths)):
    # print(centerline_paths[idx].split('/')[-1].split('.')[0].split('_')[0])
    assert image_paths[idx].split('/')[-1].split('.')[0] == label_paths[idx].split('/')[-1].split('.')[0].split('_m')[0]
    assert  image_paths[idx].split('/')[-1].split('.')[0] == centerline_paths[idx].split('/')[-1].split('.')[0].split('_m')[0]
    assert image_paths[idx].split('/')[-1].split('.')[0] == Liver_paths[idx].split('/')[-1].split('.')[0].split('_m')[0]
    image_name = str(centerline_paths[idx].split('/')[-1].split('.')[0].split('_m')[0])
    # print(image_name)

    ct = sitk.ReadImage(image_paths[idx],sitk.sitkFloat32)  # sitk.sitkInt16 Read one image using SimpleITK
    origin = ct.GetOrigin()
    direction = ct.GetDirection()
    ct_array = sitk.GetArrayFromImage(ct)
    seg = sitk.ReadImage(label_paths[idx], sitk.sitkFloat32)
    seg_array = sitk.GetArrayFromImage(seg)

    Cline = sitk.ReadImage(centerline_paths[idx], sitk.sitkFloat32)
    Cline_array = sitk.GetArrayFromImage(Cline)

    liver = sitk.ReadImage(Liver_paths[idx], sitk.sitkFloat32)
    liver_array = sitk.GetArrayFromImage(liver)
    print('-------', 'image_name', '-------')
    print('original space', np.array(ct.GetSpacing()))
    print('original shape', ct_array.shape)

    # step1: spacing interpolation

    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / config['xyz_thickness'][-1],
                                       ct.GetSpacing()[0] / config['xyz_thickness'][0],
                                       ct.GetSpacing()[1] / config['xyz_thickness'][1]), order=3)
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / config['xyz_thickness'][-1],
                                        ct.GetSpacing()[0] / config['xyz_thickness'][0],
                                        ct.GetSpacing()[1] / config['xyz_thickness'][1]), order=0)
    Cline_array = ndimage.zoom(Cline_array, (ct.GetSpacing()[-1] / config['xyz_thickness'][-1],
                                         ct.GetSpacing()[0] / config['xyz_thickness'][0],
                                         ct.GetSpacing()[1] / config['xyz_thickness'][1]), order=0)
    liver_array = ndimage.zoom(liver_array, (ct.GetSpacing()[-1] / config['xyz_thickness'][-1],
                                         ct.GetSpacing()[0] / config['xyz_thickness'][0],
                                         ct.GetSpacing()[1] / config['xyz_thickness'][1]), order=0)
    print('new space', config['xyz_thickness'])
    print('zoomed shape:', ct_array.shape, ',', seg_array.shape)

    print(seg_array.shape)
    # z = np.any(liver_array, axis=(1, 2))  # seg_array.shape(125, 256, 256)
    # start_slice, end_slice = np.where(z)[0][[0, -1]]
    # if start_slice - config['expand_slice'] < 0:
    #     start_slice = 0
    # else:
    #     start_slice -= config['expand_slice']
    # if end_slice + config['expand_slice'] >= liver_array.shape[0]:
    #     end_slice = liver_array.shape[0] - 1
    # else:
    #     end_slice += config['expand_slice']
    # ct_array = ct_array[start_slice:end_slice + 1, :, :]
    # seg_array = seg_array[start_slice:end_slice + 1, :, :]
    # Cline_array = Cline_array[start_slice:end_slice + 1, :, :]
    # liver_array = liver_array[start_slice:end_slice + 1, :, :]

    print('effective shape:', ct_array.shape, ',', seg_array.shape, ',', Cline_array.shape, ',', liver_array.shape)
    save_path  = '/media/DataA/LiverVessel/dataset/resampled_public_data/'
    saved_preprocessed(ct_array, origin, direction, config['xyz_thickness'], save_path + 'ct/' + image_name + '.nii.gz')
    saved_preprocessed(seg_array, origin, direction, config['xyz_thickness'], save_path + 'label/' + image_name + '_mask.nii.gz')
    saved_preprocessed(Cline_array, origin, direction, config['xyz_thickness'], save_path + 'cline/' + image_name + '_maskcl.nii.gz')
    saved_preprocessed(liver_array, origin, direction, config['xyz_thickness'], save_path + 'liver_mask/' + image_name + '_liver.nii.gz')