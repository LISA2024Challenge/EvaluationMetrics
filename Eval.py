#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from evalutils.io import SimpleITKLoader

import SimpleITK
import numpy as np
import pandas as pd


class Metrics():
    def score(self, ground_truth_path, predicted_path):
        loader = SimpleITKLoader()
        gt = loader.load_image(ground_truth_path)
        pred = loader.load_image(predicted_path)

        caster = SimpleITK.CastImageFilter()
        caster.SetOutputPixelType(SimpleITK.sitkFloat32)
        caster.SetNumberOfThreads(1)

        gt = caster.Execute(gt)
        pred = caster.Execute(pred)

        dsc_value = self.DSC(gt, pred)
        hd_value = self.HD(gt, pred)
        hd95_value = self.HD95(gt, pred)
        assd_value = self.ASSD(gt, pred)
        rve_value = self.RVE(gt, pred)
        f1_value = self.F1(gt, pred)
        f2_value = self.F2(gt, pred)
        precision_value = self.Precision(gt, pred)
        recall_value = self.Recall(gt, pred)

        return {
            'DSC': dsc_value,
            'HD': hd_value,
            'HD95': hd95_value,
            'ASSD': assd_value,
            'RVE': rve_value,
            'F1': f1_value,
            'F2': f2_value,
            'Precision': precision_value,
            'Recall': recall_value
        }

    def DSC(self, gt, pred):
        gt_flat = gt.flatten()
        pred_flat = pred.flatten()

        intersection = sum(pred_flat * gt_flat)
        dice_coefficient = (2.0 * intersection) / (sum(pred_flat) + sum(gt_flat))

        return dice_coefficient


    def HD(self, gt, pred):
        hausdorff_filter = SimpleITK.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(gt, pred)
        hausdorff_distance = hausdorff_filter.GetHausdorffDistance()

        return hausdorff_distance


    def HD95(self, gt, pred):
        hausdorff_filter = SimpleITK.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(gt, pred)

        distance_map = hausdorff_filter.GetDistanceMap()
        distances = SimpleITK.GetArrayFromImage(distance_map)

        sorted_distances = np.sort(distances.flatten())
        idx_95_percentile = int(0.95 * len(sorted_distances))
        hd95_distance = sorted_distances[idx_95_percentile]

        return hd95_distance

    def ASSD(self, gt, pred):
        surface_distance_map = SimpleITK.Abs(SimpleITK.SignedMaurerDistanceMap(gt) - SimpleITK.SignedMaurerDistanceMap(pred))
        assd_distance = surface_distance_map.GetArrayView().mean()

        return assd_distance

    def RVE(self, gt, pred):
        gt_volume = np.sum(gt)
        pred_volume = np.sum(pred)
        rve_value = abs((pred_volume - gt_volume) / gt_volume)

        return rve_value

    def F1(self, gt, pred):
        intersection = np.sum(np.logical_and(gt, pred))
        precision = intersection / np.sum(pred)
        recall = intersection / np.sum(gt)
        f1_value = 2 * (precision * recall) / (precision + recall + 1e-10)

        return f1_value

    def F2(self, gt, pred):
        beta = 2  # You can adjust the beta value based on your preference
        intersection = np.sum(np.logical_and(gt, pred))
        precision = intersection / np.sum(pred)
        recall = intersection / np.sum(gt)
        f2_value = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-10)

        return f2_value

    def Precision(self, gt, pred):
        precision = np.sum(np.logical_and(gt, pred)) / np.sum(pred)

        return precision

    def Recall(self, gt, pred):
        recall = np.sum(np.logical_and(gt, pred)) / np.sum(gt)

        return recall


if __name__=='__main__':
    metrics = ImageMetrics()
    output_excel_path = "metrics.xlsx"
    metrics_df = pd.DataFrame(columns=['Filename', 'DSC', 'HD', 'HD95', 'ASSD', 'RVE', 'F1', 'F2', 'Precision', 'Recall'])

    gt_dir = "GroundTruths"
    pred_dir = "Predictions"

    gt_files = [file for file in os.listdir(gt_dir)]

    max_values = {
        'DSC': 0, 'HD': 256, 'HD95': 256, 'ASSD': 1000, 'RVE': 1, 'F1': 0, 'F2': 0, 'Precision': 0, 'Recall': 0
                    }

    for file in gt_files:
        patient_id = file.split("_")[0]

        gt_file_path = os.path.join(gt_path, file)
        prediction_file_path = os.path.join(pred_dir, f'{patient_id}_prediction.nii.gz')

        if os.path.exists(gt_file_path) and os.path.exists(prediction_file_path):
            metrics_dict = metrics.score(gt_file_path, prediction_file_path)
            metrics_dict['Filename'] = patient_id
            metrics_df = metrics_df.append(metrics_dict, ignore_index=True)
        else:
            metrics_dict = max_values
            metrics_dict['Filename'] = patient_id
            metrics_df = metrics_dict.append(metrics_dict, ignore_index=True)

    metrics_df.to_excel(output_excel_path, index=False)
