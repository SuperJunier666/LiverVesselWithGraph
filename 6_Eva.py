import torch
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial import cKDTree
from monai.metrics import HausdorffDistanceMetric

def dice_score(prediction, target):
    smooth = 1e-5
    num_classes = target.size(0)
    prediction = prediction.view(num_classes, -1)
    target = target.view(num_classes, -1)

    intersection = (prediction * target)

    dice = (2. * intersection.sum(1) + smooth) / (prediction.sum(1) + target.sum(1) + smooth)

    return dice

def hausdorff_95(submission, groundtruth, spacing=np.array([1.6, 0.739, 0.739])):
    # There are more efficient algorithms for hausdorff distance than brute force, however, brute force is sufficient for datasets of this size.
    submission_points = spacing * np.array(np.where(submission), dtype=np.uint16).T
    submission_kdtree = cKDTree(submission_points)

    groundtruth_points = spacing * np.array(np.where(groundtruth), dtype=np.uint16).T
    groundtruth_kdtree = cKDTree(groundtruth_points)

    distances1, _ = submission_kdtree.query(groundtruth_points)
    distances2, _ = groundtruth_kdtree.query(submission_points)
    return max(np.quantile(distances1, 0.95), np.quantile(distances2, 0.95))

if __name__ == '__main__':

    # LabelDir = '/media/Data/yanxc/task1/graph_projtct/outputs/other_methods/withT/label/'
    # PredDir = '/media/Data/yanxc/task1/graph_projtct/outputs/other_methods/withT/chap3'

    # LabelDir = '/media/Data/yanxc/task1/test_dataset/private/label_noTrunk/'
    LabelDir = '/media/Data/yanxc/task1/graph_projtct/outputs/other_methods/noT2/label_noTrunk/'
    PredDir = '/media/Data/yanxc/task1/graph_projtct/outputs/other_methods/noT2/chap3/'
    # PredDir = '/media/Data/yanxc/task1/graph_projtct/outputs/other_methods/noT/UNet'

    Architectures = ['chap3']
    architecture = Architectures[-1]

    ############################
    # load data
    ###########################
    label_paths = [os.path.join(LabelDir, x)
                for x in os.listdir(LabelDir)
                if x.endswith('.nii.gz')]

    pred_paths = [os.path.join(PredDir, x)
                   for x in os.listdir(PredDir)
                   if x.endswith('.nii.gz')]
    label_paths.sort()
    pred_paths.sort()
    print(label_paths)
    print(pred_paths)

    # for idx in range(len(pred_paths)):
    #     img = nib.load(pred_paths[idx])
    #     image_name = str(pred_paths[idx].split('/')[-1].split('.')[0].split('_')[0])
    #     qform = img.get_qform()
    #     img.set_qform(qform)
    #     sfrom = img.get_sform()
    #     img.set_sform(sfrom)
    #     nib.save(img, pred_paths[idx])
    #     print(image_name)
    # print('No data errors')
    # assert len(label_paths) == len(pred_paths)


    results = {'Dice': [], 'Hd95': [],'Acc': [],'Recall': [],'Pre': [],'Sen': [],'Spe': [],'F1': []}
    for index in range(len(label_paths)):
        print(index,label_paths[index].split('/')[-1].split('.')[0].split('_m')[0],pred_paths[index].split('/')[-1].split('.')[0].split('_v')[0])
        # assert label_paths[index].split('/')[-1].split('.')[0].split('_m')[0]== pred_paths[index].split('/')[-1].split('.')[0].split('_v')[0]

        label_itk = nib.load(label_paths[index])
        pred_itk = nib.load(pred_paths[index])

        label_array = label_itk.get_fdata()
        affine = label_itk.affine
        spacing = [-affine[0][0],-affine[1][1],affine[2][2]]
        pred_array = pred_itk.get_fdata()

        label_tensor = torch.from_numpy(label_array).long()
        output_tensor = torch.from_numpy(pred_array).long()
        label_onehot = torch.nn.functional.one_hot(label_tensor, num_classes=2).permute(3, 0, 1, 2).contiguous()
        output_onehot = torch.nn.functional.one_hot(output_tensor, num_classes=2).permute(3, 0, 1, 2).contiguous()

        dices = dice_score(prediction=output_onehot, target=label_onehot).numpy()
        hd95 = hausdorff_95(pred_array,label_array,spacing=spacing)


        TP = np.sum(pred_array +label_array)
        TN = np.sum((1-pred_array) * (1-label_array))
        FP = np.sum(pred_array * (1-label_array))
        FN = np.sum((1-pred_array) * label_array)

        Acc = (TP + TN + 10e-4)/(TP + FN + TN + FP + 10e-4)
        Recall = (TP+ 10e-4) / (TP + FN + 10e-4)
        Pre = (TP + 10e-4) / (TP + FP + 10e-4)
        Sen = (TP + 10e-4) /(TP + FN + 10e-4)
        Spe = (TN + 10e-4) /(TN + FP + 10e-4)
        F1 = (2 * Pre * Recall) / (Pre + Recall)
        print(dices[1],hd95,F1)

        results['Dice'].append(dices[1])
        results['Hd95'].append(hd95)
        results['Acc'].append(Acc)
        results['Recall'].append(Recall)
        results['Pre'].append(Pre)
        results['Sen'].append(Sen)
        results['Spe'].append(Spe)
        results['F1'].append(F1)

    print(np.mean(results['Dice'][1:]))

    data_frame = pd.DataFrame(
        data={
              'Dice': results['Dice'],
              'Hd95': results['Hd95'],
              'Acc': results['Acc'],
               'Recall' : results['Recall'],
                'Pre':results['Pre'],
                'Sen':results['Sen'],
                'Spe': results['Spe'],
                'F1': results['F1'],
              },
        index=range(1, len(label_paths) + 1))

    data_frame.to_csv('outputs/other_methods' + '/' + 'Eva_Result_' + architecture + '.csv',index_label='Fusion')


