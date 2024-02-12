
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code presents metrics for segmentation evaluation. The metrics are:
1. Dice Similarity Coefficient (DSC)
2. Hausdorff Distance (HD) and 95 Hausdorff Distance (95HD)
3. Average Symmetric Surface Distance (ASSD), 
4. Relative Volume Error (RVE)
"""
import os
from evalutils.io import SimpleITKLoader
import SimpleITK as sitk
import numpy as np
import pandas as pd
from surface_distance.metrics import compute_surface_distances, compute_average_surface_distance, compute_dice_coefficient, compute_robust_hausdorff
def get_maps(binary_seg):
    dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(binary_seg,
                           squaredDistance=False,
                           useImageSpacing=True))
    surface_map = sitk.LabelContour(binary_seg)

    return (dist_map, surface_map)

class Metrics():
    def score(self, ground_truth_path, predicted_path, voxel_sz):
        loader = SimpleITKLoader()
        gt = loader.load_image(ground_truth_path)
        pred = loader.load_image(predicted_path)

        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkFloat32)
        caster.SetNumberOfThreads(1)

        gt = caster.Execute(gt)
        pred = caster.Execute(pred)

        dsc_value = self.DSC(gt, pred)
        hd_value = self.HD(gt, pred, voxel_sz)
        hd95_value = self.HD95(gt, pred, voxel_sz)
        assd_value = self.ASSD(gt, pred)
        rve_value = self.RVE(gt, pred)

        return dsc_value, hd_value, hd95_value, assd_value, rve_value
    
    def DSC(self, gt, pred):
        gt = sitk.GetArrayFromImage(gt)
        pred = sitk.GetArrayFromImage(pred)
        gt = gt.astype(dtype=bool)
        pred = pred.astype(dtype=bool)
        dice_coeff = compute_dice_coefficient(gt, pred)
        return dice_coeff


    def HD(self, gt, pred, voxel_sz):
        gt = sitk.GetArrayFromImage(gt)
        pred = sitk.GetArrayFromImage(pred)
        gt = gt.astype(dtype=bool)
        pred = pred.astype(dtype=bool)
        surface_dist = compute_surface_distances(pred, gt, spacing_mm=voxel_sz)
        Haus = compute_robust_hausdorff(surface_dist, 100)
        return Haus

    def HD95(self, gt, pred, voxel_sz):
        gt = sitk.GetArrayFromImage(gt)
        pred = sitk.GetArrayFromImage(pred)
        gt = gt.astype(dtype=bool)
        pred = pred.astype(dtype=bool)
        surface_dist = compute_surface_distances(pred, gt, spacing_mm=voxel_sz)
        Haus_95 = compute_robust_hausdorff(surface_dist, 95)
        return Haus_95

    def ASSD(self, gt, pred):
        gt = sitk.GetArrayFromImage(gt)
        pred = sitk.GetArrayFromImage(pred)
        gt = gt.astype(dtype=bool)
        pred = pred.astype(dtype=bool)
        surface_dist = compute_surface_distances(pred, gt, spacing_mm=[1,1,1])
        mean_surface_dis = compute_average_surface_distance(surface_dist)
        return np.mean(mean_surface_dis)
    
    
    def RVE(self, gt, pred): # Relative Volume Error (RVE)
        gt = sitk.GetArrayFromImage(gt)
        pred = sitk.GetArrayFromImage(pred)
        gt_flat = gt.flatten()
        pred_flat = pred.flatten()
        
        gt_volume = np.sum(gt_flat)
        pred_volume = np.sum(pred_flat)
        rve_value = abs((pred_volume - gt_volume) / gt_volume)

        return rve_value
    

if __name__=='__main__':
    metrics = Metrics() 
    output_excel_path = "E:\\Data\\CHLA\\LISA\\LISA_evaluation_metrics\\metrics_seg.xlsx"
    voxel_sz = [1,1,1]
    metrics_df = pd.DataFrame(columns=['patient_id', 'DSC', 'HD', 'HD95', 'ASSD'])
    rows=[]
    gt_dir = "E:\\Data\\CHLA\\LISA\\LISA_evaluation_metrics\\GroundTruths\\"
    pred_dir = "E:\\Data\\CHLA\\LISA\\LISA_evaluation_metrics\\Predictions\\"

    gt_files =[file for file in os.listdir(gt_dir) if file.endswith('.nii.gz')]
    pred_files = [file for file in os.listdir(pred_dir) if file.endswith('.nii.gz')]
    
    Default_values = {
        'Patient_ID': None, 'DSC': 0, 'HD': 256, 'HD95': 256, 'ASSD': 1000 , 'RVE': 1}

    average = lambda x, y: (x + y) / 2
    dsc_value_R, hd_value_R, hd95_value_R, assd_value_R, rve_value_R = 0, 256, 256, 1000, 1
    dsc_value_L, hd_value_L, hd95_value_L, assd_value_L, rve_value_L = 0, 256, 256, 1000, 1

    for ind in range(0,len(gt_files)):

        gt_file_path = os.path.join(gt_dir, gt_files[ind]) #gt_path -> gt_dir
        prediction_file_path = os.path.join(pred_dir, pred_files[ind])
        
        if os.path.exists(gt_file_path) and os.path.exists(prediction_file_path):
            if 'R' in gt_file_path:
                dsc_value_R, hd_value_R, hd95_value_R, assd_value_R, rve_value_R = metrics.score(gt_file_path, prediction_file_path, voxel_sz)
            if 'L' in gt_file_path:
                dsc_value_L, hd_value_L, hd95_value_L, assd_value_L, rve_value_L = metrics.score(gt_file_path, prediction_file_path, voxel_sz)

            dsc_value = average(dsc_value_R, dsc_value_L)
            hd_value = average(hd_value_R, hd_value_L)
            hd95_value = average(hd95_value_R, hd95_value_L)
            assd_value = average(assd_value_R, assd_value_L)
            rve_value = average(rve_value_R, rve_value_L)

            rows.append({'Patient_ID': gt_files[ind], 'DSC': dsc_value, 'HD': hd_value, 'HD95': hd95_value, 'ASSD': assd_value, 'RVE': rve_value})
          
        else:
            rows.append(Default_values)
            Default_values['Patient_ID'] = gt_files[ind]
    
    metrics_df = pd.DataFrame(rows)  

    metrics_df.to_excel(output_excel_path, index=False)