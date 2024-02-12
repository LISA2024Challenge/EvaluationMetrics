
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This code presents metrics for classification evaluation. The metrics are:
1. F1 score, 
2. F2 score,
3. precision, and
4. recall
""" 
import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

def get_maps(binary_seg):
    dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(binary_seg,
                           squaredDistance=False,
                           useImageSpacing=True))
    surface_map = sitk.LabelContour(binary_seg)

    return (dist_map, surface_map)

class Metrics():
    def score(self, ground_truth_path, predicted_path):
        gt = np.load(ground_truth_path)
        pred = np.load(predicted_path)

        f1_value = self.F1(gt, pred)
        f2_value = self.F2(gt, pred)
        precision_value = self.Precision(gt, pred)
        recall_value = self.Recall(gt, pred)
        accuracy_value = self.Accuracy(gt, pred)
        return f1_value, f2_value, precision_value, recall_value, accuracy_value
    

    def F1(self, gt, pred):
        f1_value = f1_score(gt, pred, average='weighted')
        return f1_value


    def F2(self, gt, pred):
        f2_value = fbeta_score(gt, pred, beta=2, average='weighted')
        return f2_value


    def Precision(self, gt, pred):
       
        precision = precision_score(gt, pred, average='weighted')
 
        return precision


    def Recall(self, gt, pred):

        recall = recall_score(gt, pred, average='weighted')

        return recall
    
    
    def Accuracy(self, gt, pred):

        accuracy = accuracy_score(gt, pred)

        return accuracy
    

if __name__=='__main__':
    metrics = Metrics() 
    output_excel_path = "E:\\Data\\CHLA\\LISA\\LISA_evaluation_metrics\\metrics_class.xlsx"
    
    metrics_df = pd.DataFrame(columns=['patient_id', 'F1', 'F2', 'Precision', 'Recall', 'Accuracy'])
    rows=[]
    gt_dir = "E:\\Data\\CHLA\\LISA\\LISA_evaluation_metrics\\GroundTruths/"
    pred_dir = "E:\\Data\\CHLA\\LISA\\LISA_evaluation_metrics\\Predictions/"
    gt_files=[]
    pred_files=[]
    for filename in os.listdir(gt_dir):
        if filename.endswith('.npy'):
           gt_files.append(filename)
           
    for filename in os.listdir(pred_dir):
        if filename.endswith('.npy'):
           pred_files.append(filename)
           
    Default_values = {
        'Patient_ID': None, 'F1': 0, 'F2': 0, 'Precision': 0, 'Recall': 0, 'Accuracy': 0}
    
    for ind in range(0,len(gt_files)):

        gt_file_path = os.path.join(gt_dir, gt_files[ind]) #gt_path -> gt_dir
        prediction_file_path = os.path.join(pred_dir, pred_files[ind])
        
        if os.path.exists(gt_file_path) and os.path.exists(prediction_file_path):
            f1_value, f2_value, precision_value, recall_value, accuracy_value = metrics.score(gt_file_path, prediction_file_path)
            # metrics_dict['Filename'] = patient_id
            # Fill the dictionary with the computed metrics
            rows.append({'Patient_ID': gt_files[ind], 'F1': f1_value, 'F2': f2_value, 'Precision': precision_value, 'Recall': recall_value, 'Accuracy': accuracy_value})
          
        else:
            rows.append(Default_values)
            Default_values['Patient_ID'] = gt_files[ind]
    
    metrics_df = pd.DataFrame(rows)  

    metrics_df.to_excel(output_excel_path, index=False)