import os
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
from utils import *
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion,generate_binary_structure
import scipy.ndimage as ndimage
import warnings

label_dir = '/media/DataA/LiverVessel/graph_projtct/outputs/MSENet2/'
cline_dir = '/media/DataA/LiverVessel/graph_projtct/outputs/Cline2/'


label_paths = [os.path.join(label_dir, x)
                    for x in os.listdir(label_dir)
                    if x.endswith('.nii.gz')]

cline_paths = [os.path.join(cline_dir, x)
                    for x in os.listdir(cline_dir)
                    if x.endswith('.nii.gz')]

label_paths.sort()
cline_paths.sort()

print(len(label_paths),label_paths)
print(len(cline_paths),cline_paths)


for idx in range(len(label_paths)):
    label_name = str(label_paths[idx].split('/')[-1].split('.')[0].split('_m')[0])
    label_itk = nib.load(label_paths[idx])
    Cline_itk = nib.load(cline_paths[idx])

    image_affine = label_itk.affine
    label = label_itk.get_fdata()
    Cline = Cline_itk.get_fdata()
    ##Add
    new_label = label + Cline
    new_label[new_label > 1 ] = 1

    struct = generate_binary_structure(3, 3)
    Vessel = binary_dilation(new_label, structure=struct, iterations=1).astype('uint8')

    Vessel_c = Vessel.copy()
    vessel_pro = measureimg(Vessel, t_num=6)
    VesselMask = vessel_pro * Vessel_c
    VesselMask = binary_erosion(VesselMask, structure=struct, iterations=1).astype('uint8')
    # struct = generate_binary_structure(3, 3)
    # CLINE = binary_dilation(Cline, structure=struct, iterations=1).astype('uint8')
    # new_label = label + Cline
    # new_label[new_label > 1] = 1
    # vessel_pro = measureimg(new_label, t_num=6)
    # print('after', np.sum(liver))

    print(label_name)
    savedImg = nib.Nifti1Image(VesselMask, image_affine)
    nib.save(savedImg,'/media/DataA/LiverVessel/graph_projtct/outputs/final_MSEnet2/' + label_name + '.nii.gz')

    # savedLiver = nib.Nifti1Image(liver, image_affine)
    # nib.save(savedLiver, '/media/DataA/LiverVessel/graph_projtct/outputs/nnUNet/' + label_name + '_liver.nii.gz')