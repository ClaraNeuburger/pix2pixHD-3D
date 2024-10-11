import nibabel as nib
import numpy as np
import os


def extract_pixel_spacing(image_path):
    img = nib.load(image_path)
    spacing = img.header.get_zooms()
    return spacing


def count_white_pixels(image_path):
    img = nib.load(image_path)
    data = img.get_fdata()
    white_pixels = np.sum(data > 0)
    return white_pixels


def process_patient_folders(patient_list, original_image_folder, segmented_folder,output_folder):
    for patient_id, nb in patient_list:
        patient_folder_name =f"{patient_id}_CT_{nb}"
        #patient_folder_name_CT =f"sCT_epoch200_patient_{patient_id}_MR_{nb}.nii"

        original_image_path = os.path.join(original_image_folder, f"{patient_folder_name}.nii.gz")

        pixel_spacing = extract_pixel_spacing(original_image_path)
        # print(f"Pixel Spacing for {patient_id}: {pixel_spacing}")
        x = pixel_spacing[0]
        y = pixel_spacing[1]
        z = pixel_spacing[2]
        volume = x*y*z*0.001

        #folder_name = f"{patient_folder_name}.nii"
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

        # print(f"Volume of skeletal_muscle: {volume_skeletal_muscle}")
        # print(f"Volume of adipose tissue: {volume_adipose_tissue}")



        with open(output_folder, 'a') as f:
            f.write(f"Patient: {patient_id}, MR: {nb}\n")
            f.write(f"Volume skeletal_muscle: {volume_skeletal_muscle}\n")
            f.write(f"Volume adipose tissue: {volume_subcutaneous_fat}\n")
            f.write(f"Volume torso fat: {volume_torso_fat}\n")
            f.write("######################################################\n")

# patient_list = [
#     ("401741594", "3"),
#     ("401397918", "2"),
#     ("401391227", "2"),
#     ("401378138", "3"),
#     ("401735368", "3"),
#     ("401398248", "2"),
#     ("401413899", "3"),
#     ("401383507", "3"),
#     ("401390765", "3"),
#     ("401413899", "2")
# ]
patient_list = [
    ("401413899", "3"),
]
# original_image_folder = '/home/radiology/Clara_intern/DataBaseWB_NII_Pix2Pix/CT'
# segmented_folder = '/home/radiology/Clara_intern/DataBaseWB_NII_Pix2Pix/Totalsegmentator_sCT'
# output_folder = "Totalsegmentator_sCT.txt"
original_image_folder = '/home/radiology/Clara_intern/DataBaseWB_NII_Pix2Pix/CT'
segmented_folder = '/home/radiology/Desktop/results_totalseg_fat/'
output_folder = "Totalsegmentator_fatMRI.txt"
process_patient_folders(patient_list, original_image_folder, segmented_folder,output_folder)


