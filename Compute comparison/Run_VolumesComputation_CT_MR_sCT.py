import nibabel as nib
import numpy as np
import os
from options.train_options import TrainOptions
import shutil


#######################################################################################################################
# Code that will compute the volumes for each tissue from the TotalSegmentator results of the sCT
#######################################################################################################################


def extract_pixel_spacing(image_path):
    img = nib.load(image_path)
    spacing = img.header.get_zooms()
    return spacing


def count_white_pixels(image_path):
    img = nib.load(image_path)
    data = img.get_fdata()
    white_pixels = np.sum(data > 0)
    return white_pixels


def process_patient_folders_sCT(patient_list, original_image_folder, segmented_folder,output_folder,epoch):
    for patient_id, nb in patient_list:
        patient_folder_name_CT =f"{patient_id}_CT_{nb}"

        original_image_path = os.path.join(original_image_folder, f"{patient_folder_name_CT}.nii.gz")

        pixel_spacing = extract_pixel_spacing(original_image_path)
        x = pixel_spacing[0]
        y = pixel_spacing[1]
        z = pixel_spacing[2]
        volume = x*y*z*0.001


        folder_name = f"sCT_epoch{epoch}_patient_{patient_id}_MR_{nb}"
        patient_segmented_folder = os.path.join(segmented_folder, folder_name)

        skeletal_muscle_path = os.path.join(patient_segmented_folder, "skeletal_muscle.nii.gz")
        subcutaneous_fat_path = os.path.join(patient_segmented_folder, "subcutaneous_fat.nii.gz")
        torso_fat_path = os.path.join(patient_segmented_folder, "torso_fat.nii.gz")

        white_pixels_skeletal_muscle = count_white_pixels(skeletal_muscle_path)
        white_pixels_subcutaneous_fat = count_white_pixels(subcutaneous_fat_path)
        white_pixels_torso_fat = count_white_pixels(torso_fat_path)

        volume_skeletal_muscle = volume * white_pixels_skeletal_muscle
        volume_subcutaneous_fat = volume*white_pixels_subcutaneous_fat
        volume_torso_fat = volume*white_pixels_torso_fat





        with open(output_folder, 'a') as f:
            f.write(f"Patient: {patient_id}, MR: {nb}\n")
            f.write(f"Volume skeletal_muscle: {volume_skeletal_muscle}\n")
            f.write(f"Volume adipose tissue: {volume_subcutaneous_fat}\n")
            f.write(f"Volume torso fat: {volume_torso_fat}\n")
            f.write("######################################################\n")


def process_patient_folders(patient_list, original_image_folder, segmented_folder,output_folder,type):
    for patient_id, nb in patient_list:
        patient_folder_name_ref =f"{patient_id}_CT_{nb}"
        if type == "MR":
            patient_folder_name =f"{patient_id}_MR_{nb}.nii"

        if type == "CT":
            patient_folder_name =f"{patient_id}_CT_{nb}.nii"

        original_image_path = os.path.join(original_image_folder, f"{patient_folder_name_ref}.nii.gz")

        pixel_spacing = extract_pixel_spacing(original_image_path)

        x = pixel_spacing[0]
        y = pixel_spacing[1]
        z = pixel_spacing[2]
        volume = x*y*z*0.001

        folder_name = f"{patient_folder_name}"
        patient_segmented_folder = os.path.join(segmented_folder, folder_name)

        skeletal_muscle_path = os.path.join(patient_segmented_folder, "skeletal_muscle.nii.gz")
        subcutaneous_fat_path = os.path.join(patient_segmented_folder, "subcutaneous_fat.nii.gz")
        torso_fat_path = os.path.join(patient_segmented_folder, "torso_fat.nii.gz")

        white_pixels_skeletal_muscle = count_white_pixels(skeletal_muscle_path)
        white_pixels_subcutaneous_fat = count_white_pixels(subcutaneous_fat_path)
        white_pixels_torso_fat = count_white_pixels(torso_fat_path)

        volume_skeletal_muscle = volume * white_pixels_skeletal_muscle
        volume_subcutaneous_fat = volume*white_pixels_subcutaneous_fat
        volume_torso_fat = volume*white_pixels_torso_fat





        with open(output_folder, 'a') as f:
            f.write(f"Patient: {patient_id}, MR: {nb}\n")
            f.write(f"Volume skeletal_muscle: {volume_skeletal_muscle}\n")
            f.write(f"Volume adipose tissue: {volume_subcutaneous_fat}\n")
            f.write(f"Volume torso fat: {volume_torso_fat}\n")
            f.write("######################################################\n")


def extract_patient_nb(txt_file_path):
    patient_nb_list = []
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            patient_id, scan_number = line.strip().split("_")
            patient_nb_list.append((patient_id, scan_number))
    return patient_nb_list


input_epoch = input("What is the epoch number of the model? ")

opt = TrainOptions().parse()

title = opt.name

path_test = f"../checkpoints/{title}/test_subjects_{title}.txt"
patient_list = extract_patient_nb(path_test)


original_image_folder = f'{opt.dataroot}/CT'


sCT_folder = f"../checkpoints/{title}/TotalSegmentator_Results/TotalSegmentator_sCT_{title}"
MR_folder = f'{opt.dataroot}/Totalsegmentator_MR'
CT_folder = f'{opt.dataroot}/Totalsegmentator_CT'

output_folder_sCT = f"../checkpoints/{title}/TotalSegmentator_Results/TotalSegmentator_sCT_{title}.txt"
output_folder_MR = f"../checkpoints/{title}/TotalSegmentator_Results/TotalSegmentator_MR_{title}.txt"
output_folder_CT = f"../checkpoints/{title}/TotalSegmentator_Results/TotalSegmentator_CT_{title}.txt"
folder_path = os.path.dirname(output_folder_sCT)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

process_patient_folders_sCT(patient_list, original_image_folder, sCT_folder,output_folder_sCT,input_epoch)
process_patient_folders(patient_list,original_image_folder, MR_folder,output_folder_MR,"MR")
process_patient_folders(patient_list,original_image_folder, CT_folder,output_folder_CT,"CT")




def delete_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Folder '{path}' deleted successfully.")
    else:
        print(f"Path '{path}' does not exist.")

delete_path(f"checkpoints")