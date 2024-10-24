import pandas as pd
import os
import shutil
import numpy as np
from pyCompare import blandAltman
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from options.train_options import TrainOptions


#######################################################################################################################
# Code that produces a txt file containing the comparison of the sCT performance vs the MRI segmentation performance
#######################################################################################################################



def read_excel_data(file_path, sheet_name, row_index, start_col, end_col):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    row_data = df.iloc[row_index, start_col:end_col].tolist()

    if len(row_data) >= 3:
        sum_last_three = sum(row_data[-3:])
    else:
        sum_last_three = 0

    row_data.append(sum_last_three)
    return row_data


def write_comparison_to_txt(file1_data, file2_data,mri_data, column_names, output_file, patient_id, modality, volumes=None):
    with open(output_file, 'a') as f:
        f.write('\n')
        f.write("#" * 90 + "\n")
        f.write(f"Patient ID: {patient_id} - Block: {modality}\n")
        f.write(f"{'Total (mL)':<30} {'Real CT':<30} {'sCT':<30} {'MRI':<30}\n")
        f.write("=" * 120 + "\n")
        for col_name, val1, val2, val3 in zip(column_names, file1_data, file2_data,mri_data):
            f.write(f"{col_name:<30} {val1:<30} {val2:<30} {val3:<30}\n")


        f.write("\n\n")


def compute_statistics_sCT(sct_values, real_ct_values, tissue_types):
    sct_values = np.array(sct_values)
    real_ct_values = np.array(real_ct_values)

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.where(real_ct_values != 0, 100 * (sct_values - real_ct_values) / real_ct_values, np.nan)

    valid_indices = np.isfinite(real_ct_values) & (real_ct_values > 0)
    if np.any(valid_indices):
        mpe = mean_absolute_percentage_error(real_ct_values[valid_indices], sct_values[valid_indices])
    else:
        mpe = np.nan

    mae = mean_absolute_error(real_ct_values, sct_values)
    rmse = np.sqrt(np.mean((sct_values - real_ct_values) ** 2))

    if np.std(real_ct_values) > 0 and np.std(sct_values) > 0:
        correlation = np.corrcoef(sct_values, real_ct_values)[0, 1]
    else:
        correlation = np.nan

    relative_errors = np.nan_to_num(relative_errors, nan=0.0)

    return {
        "mae": mae,
        "mpe": mpe,
        "rmse": rmse,
        "correlation": correlation,
        "relative_errors": relative_errors
    }


def compute_statistics_MR(mr_values, real_ct_values, tissue_types):
    sct_values = np.array(mr_values)
    real_ct_values = np.array(real_ct_values)


    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.where(real_ct_values != 0, 100 * (sct_values - real_ct_values) / real_ct_values, np.nan)

    valid_indices = np.isfinite(real_ct_values) & (real_ct_values > 0)
    if np.any(valid_indices):
        mpe = mean_absolute_percentage_error(real_ct_values[valid_indices], sct_values[valid_indices])
    else:
        mpe = np.nan

    mae = mean_absolute_error(real_ct_values, sct_values)
    rmse = np.sqrt(np.mean((sct_values - real_ct_values) ** 2))

    if np.std(real_ct_values) > 0 and np.std(sct_values) > 0:
        correlation = np.corrcoef(sct_values, real_ct_values)[0, 1]
    else:
        correlation = np.nan

    relative_errors = np.nan_to_num(relative_errors, nan=0.0)

    return {
        "mae": mae,
        "mpe": mpe,
        "rmse": rmse,
        "correlation": correlation,
        "relative_errors": relative_errors
    }


def write_statistics_to_txt(patient_id, modality, stats, tissue_types, output_file,mod):
    with open(output_file, 'a') as f:
        if mod == 'sCT':
            f.write("sCT and CT comparison:\n")
        if mod == 'MR':
            f.write("MR TotalSegmentator and CT comparison:\n")
        f.write(f"Patient ID: {patient_id} - Block: {modality} (Statistics)\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Statistic':<30} {'Value'}\n")
        f.write("=" * 90 + "\n")
        f.write(f"Mean Absolute Error (MAE): {stats['mae']:.4f}\n")
        f.write(f"Mean Percentage Error (MPE): {stats['mpe']:.2f}%\n")
        f.write(f"Root Mean Squared Error (RMSE): {stats['rmse']:.4f}\n")
        f.write(f"Pearson Correlation Coefficient: {stats['correlation']:.4f}\n")
        f.write("\n")
        f.write(f"{'Tissue Type':<30} {'Relative Error (%)'}\n")
        f.write("=" * 90 + "\n")
        for tissue, rel_err in zip(tissue_types, stats['relative_errors']):
            f.write(f"{tissue:<30} {rel_err:.2f}%\n")
        f.write("\n\n")


def parse_volume_data(volume_data_path):
    volume_data = {}

    with open(volume_data_path, 'r') as file:
        lines = file.readlines()

    current_patient_id = None
    current_mr = None

    for line in lines:
        line = line.strip()
        if line.startswith("Patient:"):
            parts = line.split(", ")
            current_patient_id = parts[0].split(": ")[1]
            current_mr = parts[1].split(": ")[1]
            volume_data[(current_patient_id, current_mr)] = {}
        elif line.startswith("Volume"):
            tissue_type, volume = line.split(": ")
            volume_data[(current_patient_id, current_mr)][tissue_type.strip()] = float(volume.strip())

    # print("Parsed Volume Data:", volume_data)

    return volume_data


def get_volumes_for_patient(volume_data, patient_id, mr_number):
    mr_number = mr_number.split('.')[0]
    key = (patient_id.strip(), mr_number.strip())

    # Retrieve the data
    if key in volume_data:
        volumes = volume_data[key]

        muscle_volume = volumes.get("Volume skeletal_muscle", 0.0)
        torso_volume = volumes.get("Volume torso fat", 0.0)
        subcutaneous_volume = volumes.get("Volume adipose tissue", 0.0)
        list_volumes = [muscle_volume,subcutaneous_volume,torso_volume]

        return list_volumes
    else:
        print(f"No data found for Patient ID: {patient_id}, MR: {mr_number}")
        return None, None, None


#######################################################################################################################

opt = TrainOptions().parse()
#title = input("What is the name of the model ou want to evaluate ? ")
plot = input('Do you want to create Bland Altman plots ? (yes/no) ')
title = opt.name

folder1 = f'../checkpoints/{title}/TotalSegmentator_Results/TotalSegmentator_sCT_{title}'
#folder2 = '/home/radiology/Clara_intern/DataBaseWB_NII_Pix2Pix/CT_BOA_results/'

output_file = f'../checkpoints/{title}/TotalSegmentator_Results/comparison_TotalSegmentator_{title}.txt'

volume_data_CT = parse_volume_data(f'../checkpoints/{title}/TotalSegmentator_Results/TotalSegmentator_CT_{title}.txt')
volume_data_sCT = parse_volume_data(f'../checkpoints/{title}/TotalSegmentator_Results/TotalSegmentator_sCT_{title}.txt')
volume_data_MR = parse_volume_data(f'../checkpoints/{title}/TotalSegmentator_Results/TotalSegmentator_MR_{title}.txt')



#######################################################################################################################

# Indications on where to read in the output.xlsx
sheet_name = 1
row_index = 16
start_col = 3
end_col = 11


column_names = ['Muscle', 'Subcutaneous Adipose Tissue','Torso fat']

cumulative_stats_sCT = {
    "mae": [],
    "mpe": [],
    "rmse": [],
    "correlation": [],
    "relative_errors": np.zeros(len(column_names))
}
cumulative_stats_MR = {
    "mae": [],
    "mpe": [],
    "rmse": [],
    "correlation": [],
    "relative_errors": np.zeros(3)
}
patient_count = 0

all_mr_values = []
all_ct_values = []
all_sct_values = []

for root, dirs, files in os.walk(folder1):
    for folder_name in dirs:
        parts = folder_name.split('_')
        patient_id = parts[3]
        instance_number = parts[5]


        ct_data = get_volumes_for_patient(volume_data_CT,str(patient_id),str(instance_number))
        sct_data = get_volumes_for_patient(volume_data_sCT,str(patient_id),str(instance_number))
        mri_data = get_volumes_for_patient(volume_data_MR,str(patient_id),str(instance_number))


        write_comparison_to_txt(ct_data, sct_data, mri_data,column_names, output_file, patient_id, instance_number)

        stats_sCT = compute_statistics_MR(ct_data, sct_data, column_names)
        stats_MR = compute_statistics_MR(ct_data,mri_data, column_names)

        all_mr_values.append(mri_data)
        all_ct_values.append(ct_data)
        all_sct_values.append(sct_data)


        write_statistics_to_txt(patient_id, instance_number, stats_sCT, column_names, output_file,'sCT')
        write_statistics_to_txt(patient_id, instance_number, stats_MR, column_names, output_file,'MR')

        cumulative_stats_sCT["mae"].append(stats_sCT["mae"])
        cumulative_stats_sCT["mpe"].append(stats_sCT["mpe"])
        cumulative_stats_sCT["rmse"].append(stats_sCT["rmse"])
        cumulative_stats_sCT["correlation"].append(stats_sCT["correlation"])
        cumulative_stats_sCT["relative_errors"] += stats_sCT["relative_errors"]

        cumulative_stats_MR["mae"].append(stats_MR["mae"])
        cumulative_stats_MR["mpe"].append(stats_MR["mpe"])
        cumulative_stats_MR["rmse"].append(stats_MR["rmse"])
        cumulative_stats_MR["correlation"].append(stats_MR["correlation"])
        cumulative_stats_MR["relative_errors"] += stats_MR["relative_errors"]

        patient_count += 1
        # print(f"Comparison and statistics appended for Patient {patient_id}, Instance {instance_number}")


mean_relative_errors_sCT = cumulative_stats_sCT["relative_errors"] / patient_count if patient_count > 0 else np.zeros(
    len(column_names))
mean_relative_errors_MR = cumulative_stats_MR["relative_errors"] / patient_count if patient_count > 0 else np.zeros(
    len(column_names))


with open(output_file, 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write("SUMMARY of Mean Statistics Across All Patients TotalSegmentator sCT vs CT\n")
    f.write("=" * 90 + "\n")
    f.write(f"Mean MAE: {np.mean(cumulative_stats_sCT['mae']):.4f}\n")
    f.write(f"Mean MPE: {np.mean(cumulative_stats_sCT['mpe']):.2f}%\n")
    f.write(f"Mean RMSE: {np.mean(cumulative_stats_sCT['rmse']):.4f}\n")
    f.write(f"Mean Pearson Correlation Coefficient: {np.nanmean(cumulative_stats_sCT['correlation']):.4f}\n")
    f.write("\n")
    f.write(f"{'Tissue Type':<30} {'Mean Relative Error (%)'}\n")
    f.write("=" * 90 + "\n")
    for tissue, mean_rel_err in zip(column_names, mean_relative_errors_sCT):
        f.write(f"{tissue:<30} {mean_rel_err:.2f}%\n")
    f.write("=" * 90 + "\n")
    f.write("\n")
    f.write("SUMMARY of Mean Statistics Across All Patients TotalSegmentator MR vs CT\n")
    f.write("=" * 90 + "\n")
    f.write(f"Mean MAE: {np.mean(cumulative_stats_MR['mae']):.4f}\n")
    f.write(f"Mean MPE: {np.mean(cumulative_stats_MR['mpe']):.2f}%\n")
    f.write(f"Mean RMSE: {np.mean(cumulative_stats_MR['rmse']):.4f}\n")
    f.write(f"Mean Pearson Correlation Coefficient: {np.nanmean(cumulative_stats_MR['correlation']):.4f}\n")
    f.write("\n")
    f.write(f"{'Tissue Type':<30} {'Mean Relative Error (%)'}\n")
    f.write("=" * 90 + "\n")
    for tissue, mean_rel_err in zip(column_names, mean_relative_errors_MR):
        f.write(f"{tissue:<30} {mean_rel_err:.2f}%\n")
    f.write("=" * 90 + "\n")

    f.write('\n')
    f.write('RESULTS AND STATISTICS for each patient of the testing set')
    f.write(content)

print("All comparisons and statistics have been processed.")


if plot == 'yes':
    path_plots = f"../checkpoints/{title}/TotalSegmentator_Results/Bland_Alman_plots"
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)
    path_mr_muslce = f"{path_plots}/MR_muscle_Bland_Alman_fig.png"
    path_sct_muscle = f"{path_plots}/sCT_muscle_Bland_Alman_fig.png"

    path_mr_adipose = f"{path_plots}/MR_adipose_Bland_Alman_fig.png"
    path_sct_adipose = f"{path_plots}/sCT_adipose_Bland_Alman_fig.png"

    path_mr_torso = f"{path_plots}/MR_torso_Bland_Alman_fig.png"
    path_sct_torso = f"{path_plots}/sCT_torso_Bland_Alman_fig.png"

    muscle_ct = [item[0] for item in all_ct_values]
    adipose_ct = [item[1] for item in all_ct_values]
    torso_ct= [item[2] for item in all_ct_values]

    muscle_sct = [item[0] for item in all_sct_values]
    adipose_sct = [item[1] for item in all_sct_values]
    torso_sct = [item[2] for item in all_sct_values]

    muscle_mr = [item[0] for item in all_mr_values]
    adipose_mr = [item[1] for item in all_mr_values]
    torso_mr = [item[2] for item in all_mr_values]


    blandAltman(muscle_mr, muscle_ct, savePath=path_mr_muslce)
    blandAltman(muscle_sct, muscle_ct, savePath=path_sct_muscle)
    blandAltman(adipose_mr, adipose_ct, savePath=path_mr_adipose)
    blandAltman(adipose_sct, adipose_ct, savePath=path_sct_adipose)
    blandAltman(torso_mr, torso_ct, savePath=path_mr_torso)
    blandAltman(torso_sct, torso_ct, savePath=path_sct_torso)



def delete_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Folder '{path}' deleted successfully.")
    else:
        print(f"Path '{path}' does not exist.")

delete_path(f"checkpoints")