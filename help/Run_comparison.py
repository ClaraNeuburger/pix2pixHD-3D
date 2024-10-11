import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

def read_excel_data(file_path, sheet_name, row_index, start_col, end_col):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    row_data = df.iloc[row_index, start_col:end_col]
    return row_data.tolist()

def write_comparison_to_txt(file1_data, file2_data, column_names, output_file, patient_id, modality):
    with open(output_file, 'a') as f:
        f.write('\n')
        f.write("#" * 90 + "\n")
        f.write(f"Patient ID: {patient_id} - Block: {modality}\n")
        f.write(f"{'Total (mL)':<30} {'sCT':<30} {'Real CT':<30}\n")
        f.write("=" * 90 + "\n")
        for col_name, val1, val2 in zip(column_names, file1_data, file2_data):
            f.write(f"{col_name:<30} {val1:<30} {val2:<30}\n")
        f.write("\n\n")

def compute_statistics(sct_values, real_ct_values, tissue_types):
    sct_values = np.array(sct_values)
    real_ct_values = np.array(real_ct_values)

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.where(real_ct_values != 0, 100 * (sct_values - real_ct_values) / real_ct_values, np.nan)

    # Mean Percentage Error (MPE) --> average percentage difference between sCT and CT for all tissue types
    valid_indices = np.isfinite(real_ct_values) & (real_ct_values > 0)
    if np.any(valid_indices):
        mpe = mean_absolute_percentage_error(real_ct_values[valid_indices], sct_values[valid_indices])
    else:
        mpe = np.nan  # Set MPE to nan if no valid data points

    # Mean Absolute Error (MAE) --> Simple, intuitive measure about how far the sCT values are from the real ones
    mae = mean_absolute_error(real_ct_values, sct_values)

    # Root Mean Square Error (RMSE) --> penalizes more larger differences (good to notice outliers)
    rmse = np.sqrt(np.mean((sct_values - real_ct_values) ** 2))

    # Pearson correlation coeff --> close to 1 means good correlation (increase or decrease together for ex)
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

def write_statistics_to_txt(patient_id, modality, stats, tissue_types, output_file):
    with open(output_file, 'a') as f:
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

#######################################################################################################################
# Replace with the correct folders :
#   folder 1 is where you keep the folders containing the results of BOA on your synthetic CTs testing set
#   folder 2 is where you keep the folders containing the results of BOA on all your real CTs
folder1 = '/home/radiology/Documents/Body-and-Organ-Analysis/data/result_sCT_BOA_3D_big_patch'
folder2 = '/home/radiology/Clara_intern/DataBaseWB_NII_Pix2Pix/CT_BOA_results/'


# Rename the output file with the name of your model
output_file = 'comparison_results_3D_big_patch_trunk.txt'

if os.path.exists(output_file):
    os.remove(output_file)

#######################################################################################################################

# Indications on where to read in the output.xlsx
sheet_name = 1
row_index = 16
start_col = 3
end_col = 11
column_names = ['Bone', 'Muscle', 'Total Adipose Tissue', 'IntraMuscular Adipose Tissue',
                'Subcutaneous Adipose Tissue', 'Visceral Adipose Tissue',
                'Perivascular Adipose Tissue', 'Epicardial Adipose Tissue']


cumulative_stats = {
    "mae": [],
    "mpe": [],
    "rmse": [],
    "correlation": [],
    "relative_errors": np.zeros(len(column_names))
}

patient_count = 0
for root, dirs, files in os.walk(folder1):
    for folder_name in dirs:
        parts = folder_name.split('_')
        patient_id = parts[3]
        instance_number = parts[5]

        matching_file = f"{patient_id}_CT_{instance_number}"
        matching_folder_path = None

        for root2, dirs2, _ in os.walk(folder2):
            for dir_name in dirs2:
                if dir_name == matching_file:
                    matching_folder_path = os.path.join(root2, dir_name)
                    break
            if matching_folder_path:
                break

        if not matching_folder_path:
            parts = folder_name.split('_')
            patient_id = parts[3]
            instance_number = parts[6]
            index = parts[4]
            alt_file = f"{patient_id}_{index}_CT_{instance_number}"
            for root2, dirs2, _ in os.walk(folder2):
                for dir_name in dirs2:
                    if dir_name == alt_file:
                        matching_folder_path = os.path.join(root2, dir_name)


        if matching_folder_path:
            file1_path = os.path.join(root, folder_name, "output.xlsx")
            file2_path = os.path.join(matching_folder_path, "output.xlsx")

            file1_data = read_excel_data(file1_path, sheet_name, row_index, start_col, end_col)
            file2_data = read_excel_data(file2_path, sheet_name, row_index, start_col, end_col)

            write_comparison_to_txt(file1_data, file2_data, column_names, output_file, patient_id, instance_number)

            stats = compute_statistics(file1_data, file2_data, column_names)
            write_statistics_to_txt(patient_id, instance_number, stats, column_names, output_file)

            cumulative_stats["mae"].append(stats["mae"])
            cumulative_stats["mpe"].append(stats["mpe"])
            cumulative_stats["rmse"].append(stats["rmse"])
            cumulative_stats["correlation"].append(stats["correlation"])
            cumulative_stats["relative_errors"] += stats["relative_errors"]

            patient_count += 1
            print(f"Comparison and statistics appended for Patient {patient_id}, Instance {instance_number}")
        else:
            print(f"No matching file found for Patient {patient_id}, Instance {instance_number}")

mean_relative_errors = cumulative_stats["relative_errors"] / patient_count if patient_count > 0 else np.zeros(len(column_names))

with open(output_file, 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write("SUMMARY of Mean Statistics Across All Patients\n")
    f.write("=" * 90 + "\n")
    f.write(f"Mean MAE: {np.mean(cumulative_stats['mae']):.4f}\n")
    f.write(f"Mean MPE: {np.mean(cumulative_stats['mpe']):.2f}%\n")
    f.write(f"Mean RMSE: {np.mean(cumulative_stats['rmse']):.4f}\n")
    f.write(f"Mean Pearson Correlation Coefficient: {np.nanmean(cumulative_stats['correlation']):.4f}\n")
    f.write("\n")
    f.write(f"{'Tissue Type':<30} {'Mean Relative Error (%)'}\n")
    f.write("=" * 90 + "\n")
    for tissue, mean_rel_err in zip(column_names, mean_relative_errors):
        f.write(f"{tissue:<30} {mean_rel_err:.2f}%\n")
    f.write("=" * 90 + "\n")
    f.write('\n')
    f.write('RESULTS AND STATISTICS for each patient of the testing set')
    f.write(content)

print("All comparisons and statistics have been processed.")
