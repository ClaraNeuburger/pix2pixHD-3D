import os
import subprocess

#######################################################################################################################
# Code to run TotalSegmentator on a folder of synthetic CTs
#######################################################################################################################

# This code needs to be run from the base (3.10 env on this computer) environment where the TotalSegmentator software
# is downloaded

title = input ('What is the name of the model you want to evaluate ?')

input_folder = f'../checkpoints/{title}/Results_{title}'
output_folder = f"../checkpoints/{title}/TotalSegmentator_Results/TotalSegmentator_sCT_{title}"



os.makedirs(output_folder, exist_ok=True)

nifti_files = [f for f in os.listdir(input_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]

for nifti_file in nifti_files:
    input_nifti_path = os.path.join(input_folder, nifti_file)

    parts = nifti_file.split('.')
    output_path = os.path.join(output_folder, parts[0])

    os.makedirs(output_path, exist_ok=True)

    cmd = [
        "TotalSegmentator",
        "-i", input_nifti_path,
        "-o", output_path,
        "--body_seg",
        "--ta", "tissue_types",
        "--statistics"
    ]

    subprocess.run(cmd)
    print(f"Processed {nifti_file}")

print(f"The resulting segmentations are available in path {output_folder}")
