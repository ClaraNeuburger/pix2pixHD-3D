from torch.nn.functional import grid_sample

from models.pix2pixHD_model import Pix2PixHDModel
import re
import torch
import torchio as tio
import glob
from options.train_options import TrainOptions
import nibabel as nib
import numpy as np
import os
import shutil
from datetime import datetime

# Function that will un-normalize the nii image back into HU
def normalize_hounsfield_units(nii_image):
    data = nii_image.get_fdata()

    # normalize to Hounsfield units
    data_hu = data * 2000 - 1000

    data_hu = np.clip(data_hu, -1000, 1000)

    hu_image = nib.Nifti1Image(data_hu, nii_image.affine, nii_image.header)

    print(f"Image normalized in Hounsfield units")
    return hu_image


# Function to recover the list of testing subjects (saved in a .txt file)
def get_subjects(path, transform=None, test_images=None):
    ct_dir = os.path.join(path, 'CT')
    mri_dir = os.path.join(path, 'MR')

    ct_paths = sorted(glob.glob(ct_dir + '/*.nii.gz'))
    mri_paths = sorted(glob.glob(mri_dir + '/*.nii.gz'))

    assert len(ct_paths) == len(mri_paths)
    subjects = []
    for (ct_path, mr_path) in zip(ct_paths, mri_paths):
        subject = tio.Subject(
            ct=tio.ScalarImage(ct_path),
            mri=tio.ScalarImage(mr_path),
        )
        subjects.append(subject)

    if test_images:
        subjects = [subject for subject in subjects if os.path.basename(subject.mri.path) in test_images]

    return tio.SubjectsDataset(subjects, transform=transform)


# def recover_mri_names(txt_file_path):
#     test_image = []
#     with open(txt_file_path, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             match = re.search(r'(\d+_MR_\d+\.nii\.gz)', line)
#             if match:
#                 test_image.append(match.group(1))
#     return test_image

def recover_mri_names(txt_file_path):
    test_image = []
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(r'(\d+_MR_\d+\.nii\.gz)', line)
            if match:
                test_image.append(match.group(1))
            else:
                match_path = re.search(r'\/[a-zA-Z0-9_/]+(\d+_MR_\d+\.nii\.gz)', line)
                if match_path:
                    test_image.append(match_path.group(1))
    return test_image


# Recreate the dataset for testing using only the testing images
class Dataset:
    def __init__(self, path, test_images=None, batch_size=1):
        self.path = path
        self.batch_size = batch_size

        validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Clamp(out_min=-1000, out_max=1000, include=['ct']),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 1000), include=['ct']),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 1000), exclude=['ct']),
        ])

        self.subjects = get_subjects(self.path, transform=None, test_images=test_images)

        self.test_set = tio.SubjectsDataset(self.subjects, transform=validation_transform)

    def get_subject_loader(self,subject):
        grid_sampler = tio.inference.GridSampler(subject,patch_size=(96, 96, 44),patch_overlap=16)
        patch_loader = torch.utils.data.DataLoader(grid_sampler,batch_size=self.batch_size)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')

        return patch_loader, aggregator


# Recover the weights of a saved model (choose the epoch and model)
def load_model(generator_path, discriminator_path, opt):
    model = Pix2PixHDModel()
    model.initialize(opt)

    state_dict_gen = torch.load(generator_path)
    model.netG.load_state_dict(state_dict_gen, strict=False)

    if os.path.exists(discriminator_path):
        state_dict_disc = torch.load(discriminator_path)
        model.netD.load_state_dict(state_dict_disc, strict=False)

    return model


# Run inference on the testing set : get the sCT slices from the trained model
# Then rebuild the 3D image as a niftii image and save it
def run_inference(model, patch_loader, aggregator, output_dir,sub):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for patches_batch in patch_loader:
            mri = patches_batch['mri'][tio.DATA].to(device).float()
            ct = patches_batch['ct'][tio.DATA].to(device)

            mri_filename = os.path.basename(patches_batch['mri']['path'][0])
            patient_number = mri_filename.split('_')[0]
            mri_number = mri_filename.split('_')[2].split('.')[0]

            locations = patches_batch[tio.LOCATION]

            output_patches = model.inference(mri, mri)  # predicted patch image

            # Add the batch's predicted patches to the aggregator
            aggregator.add_batch(output_patches, locations)

    # Get the final output tensor (reconstructed full 3D image)
    output_tensor = aggregator.get_output_tensor()
    output_tensor = output_tensor.cpu().numpy().squeeze()
    spacing = sub.ct.spacing

    affine = np.eye(4)
    affine[0, 0] = spacing[0]  # X spacing
    affine[1, 1] = spacing[1]  # Y spacing
    affine[2, 2] = spacing[2]  # Z spacing

    sCT_nifti = nib.Nifti1Image(output_tensor, affine=affine)
    sCT_nifti = normalize_hounsfield_units(sCT_nifti)
    nib.save(sCT_nifti, f'{output_dir}/sCT_epoch{epoch}_patient_{patient_number}_MR_{mri_number}.nii.gz')

    print(f"Saved original CT and sCT - {patient_number}")





# Updates the process_lib file from boa automatically with the new folder path that you created with your sCT
def update_boa_dir(parent_folder_path):
    process_lib_path = '/home/radiology/Documents/Body-and-Organ-Analysis/process_lib.sh'

    with open(process_lib_path, 'r') as file:
        script_lines = file.readlines()

    updated_lines = []
    for line in script_lines:
        if line.startswith('INPUT_DIR='):
            updated_lines.append(f'INPUT_DIR="{parent_folder_path}"\n')
        elif line.startswith('OUTPUT_DIR='):
            updated_lines.append(f'OUTPUT_DIR="{parent_folder_path}"\n')
        else:
            updated_lines.append(line)

    with open(process_lib_path, 'w') as file:
        file.writelines(updated_lines)

    print(f"Updated {process_lib_path} with new input and output directories.")


def organize_images_with_parent(source_folder, destination_folder):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parent_folder_name = f'result_sCT_BOA_real_time_{current_time}'
    parent_folder_path = os.path.join(destination_folder, parent_folder_name)

    os.makedirs(parent_folder_path, exist_ok=True)

    nii_files = [f for f in os.listdir(source_folder) if f.endswith('.nii.gz')]

    for nii_file in nii_files:
        base_name = os.path.splitext(os.path.splitext(nii_file)[0])[0]

        new_folder_path = os.path.join(parent_folder_path, base_name)
        os.makedirs(new_folder_path, exist_ok=True)

        new_file_path = os.path.join(new_folder_path, 'image.nii.gz')

        shutil.copy2(os.path.join(source_folder, nii_file), new_file_path)

        print(f"Copied {nii_file} to {new_file_path}")

    update_boa_dir(parent_folder_path)
    print('BOA process_lib path updated')



#######################################################################################################################
# Where you must change the paths for your database, model

# path to the database
path = '/home/radiology/Clara_intern/DataBaseWB_NII_Pix2Pix'

# path to the list (text file) of patients in the testing set
path_test = '/home/radiology/Clara_intern/pix2pixHD 3D + git/checkpoints/test_subjects_20241007-163138.txt'
test_images = recover_mri_names(path_test)

print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# Change to your chosen saved model
model_path_G = '/home/radiology/Clara_intern/pix2pixHD 3D + git/checkpoints/mr2ctHD/Training 3D big patch/200_net_G.pth'
model_path_D = '/home/radiology/Clara_intern/pix2pixHD 3D + git/checkpoints/mr2ctHD/Training 3D big patch/200_net_D.pth'
opt = TrainOptions().parse()
model = load_model(model_path_G,model_path_D, opt)
model.to(device)

# Epoch of chosen model (for the name of your sCT)
epoch = 200

# Create the testing set
dataset = Dataset(path, test_images=test_images, batch_size=1)
for i in range(len(test_images)-1):
    sub = dataset.subjects[i]
    patch_loader, aggregator = dataset.get_subject_loader(sub)

    # Where you want to save the sCT
    output_directory = '/home/radiology/Clara_intern/pix2pixHD 3D + git/testing_results/Results 3D big patch'

    run_inference(model, patch_loader, aggregator, output_directory, sub)

# Folder which will be used by BOA (necessary because it requires a specific folder structure)
destination_folder = '/home/radiology/Documents/Body-and-Organ-Analysis/data'
organize_images_with_parent(output_directory, destination_folder)

#######################################################################################################################