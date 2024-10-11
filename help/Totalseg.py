import os
import subprocess


input_folder = "/home/radiology/Clara_intern/pix2pixHD 3D + git/testing_results/Results 3D just trunk"
output_folder = "/home/radiology/Clara_intern/DataBaseWB_NII_Pix2Pix/Totalsegmentator_sCT"

os.makedirs(output_folder, exist_ok=True)

nifti_files = [f for f in os.listdir(input_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]

for nifti_file in nifti_files:
    input_nifti_path = os.path.join(input_folder, nifti_file)

    output_path = os.path.join(output_folder, os.path.splitext(nifti_file)[0])

    # Create the output folder (same name as input file without extension)
    os.makedirs(output_path, exist_ok=True)

    # Define the TotalSegmentator command
    cmd = [
        "TotalSegmentator",
        "-i", input_nifti_path,
        "-o", output_path,
        "--body_seg",
        "--ta", "tissue_types",
        "--statistics"
    ]

    # Run the command
    subprocess.run(cmd)
    print(f"Processed {nifti_file}")

# Print when all files are processed
print("All files processed.")
