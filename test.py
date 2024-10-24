from models.pix2pixHD_model import Pix2PixHDModel
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


# Helper functions to combine the 2 MRI dixon types
class MySubject(tio.Subject):
    def check_consistent_attribute(self, *args, **kwargs) -> None:
        kwargs['relative_tolerance'] = 1e-2
        kwargs['absolute_tolerance'] = 1e-2
        return super().check_consistent_attribute(*args, **kwargs)

# Used when you have 2 channels of MRI to normalize them independently
def load_and_transform_image(image_path, transform=None):
    image = tio.ScalarImage(image_path)
    if transform:
        image = transform(image)
    return image

# Combine the 3 MRI sequences into an image with 2 channels
def combine_mri_channels(mri_0, mri_1):
    combined_data = np.stack((mri_0.data.squeeze(), mri_1.data.squeeze()), axis=0)
    combined_mri = tio.ScalarImage(tensor=combined_data,
                                   spatial_shape=mri_0.shape[1:],
                                   affine=mri_0.affine)
    return combined_mri



# Loading the subjects from the given path
def get_subjects(path, nb_channels, test_subjects, transform=None):

    ### If the MRI image only has one channel, simply load the images
    if nb_channels == 1:
        ct_dir = os.path.join(path, 'CT')
        mri_dir = os.path.join(path, 'MR')

        ct_paths = sorted(glob.glob(ct_dir + '/*.nii.gz'))
        mri_paths = sorted(glob.glob(mri_dir + '/*.nii.gz'))

        assert len(ct_paths) == len(mri_paths)

        testing_subjects = []
        ct_paths_check = []

        for subject in test_subjects:
            ct_path_check = subject[0]
            ct_paths_check.append(ct_path_check)

        for (ct_path, mri_path) in zip(ct_paths, mri_paths):
            subject = MySubject(
                ct= tio.ScalarImage(ct_path),
                mri= tio.ScalarImage(mri_path),
            )
            if ct_path in ct_paths_check:
                testing_subjects.append(subject)


    ### If the MRI image has 2 channels aka if you use the fat and water sequence from the dixon
    if nb_channels == 2:
        ct_dir = os.path.join(path, 'CT')
        mri_W_dir = os.path.join(path, 'MR water')
        mri_F_dir = os.path.join(path, 'MR fat resized')

        ct_paths = sorted(glob.glob(ct_dir + '/*.nii.gz'))
        mri_W_paths = sorted(glob.glob(mri_W_dir + '/*.nii.gz'))
        mri_F_paths = sorted(glob.glob(mri_F_dir + '/*.nii.gz'))
        assert len(ct_paths) == len(mri_W_paths)

        testing_subjects = []
        ct_paths_check= []

        for subject in test_subjects:
            ct_path_check = subject[0]
            ct_paths_check.append(ct_path_check)

        for (ct_path, mri_W_path, mri_F_path) in zip(ct_paths, mri_W_paths, mri_F_paths):
            ct_image = load_and_transform_image(ct_path,transform=None)

            transform_mri =tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 1000))
            mri_W_image = load_and_transform_image(mri_W_path,transform=transform_mri)
            mri_F_image = load_and_transform_image(mri_F_path,transform=transform_mri)
            combined_mri = combine_mri_channels(mri_W_image,mri_F_image)

            subject = MySubject(
                ct=ct_image,
                mri=combined_mri,
            )
            if ct_path in ct_paths_check:
                testing_subjects.append(subject)

    return testing_subjects


def load_test_subject_1(test_file_path,path):
    test_subjects = set()
    with open(test_file_path, 'r') as f:
        for line in f:
            patient_id, scan_number = line.strip().split("_")
            ct_path = os.path.join(path,f"CT/{patient_id}_CT_{scan_number}.nii.gz")
            mr_path = os.path.join(path,f"MR/{patient_id}_MR.nii.gz")
            test_subjects.add((ct_path, mr_path))
    return test_subjects

def load_test_subject_2(test_file_path,path):
    test_subjects = set()
    with open(test_file_path, 'r') as f:
        for line in f:
            patient_id, scan_number = line.strip().split("_")
            ct_path = os.path.join(path,f"CT/{patient_id}_CT_{scan_number}.nii.gz")
            mr_W_path = os.path.join(path,f"MR water/{patient_id}_MR_{scan_number}.nii.gz")
            mr_F_path = os.path.join(path, f"MR fat resized/{patient_id}_MR_{scan_number}.nii.gz")
            test_subjects.add((ct_path, mr_W_path, mr_F_path))
    return test_subjects



# Recreate the dataset for testing using only the testing images
class Dataset:
    def __init__(self, path,patch_size,nb_channels, test_images, batch_size=1):
        self.path = path
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.test_images = test_images
        self.nb_channels = nb_channels

        validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Clamp(out_min=-1000, out_max=1000, include=['ct']),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 1000), include=['ct']),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 1000), exclude=['ct']),
        ])

        self.subjects = get_subjects(self.path, self.nb_channels, self.test_images, transform=None)

        self.test_set = tio.SubjectsDataset(self.subjects, transform=validation_transform)

    def get_subject_loader(self,subject):
        grid_sampler = tio.inference.GridSampler(subject,patch_size=self.patch_size,patch_overlap=16)
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

            mri_filename = os.path.basename(patches_batch['ct']['path'][0])
            patient_number = mri_filename.split('_')[0]
            mri_number = mri_filename.split('_')[2].split('.')[0]

            locations = patches_batch[tio.LOCATION]

            output_patches = model.inference(mri, mri)

            aggregator.add_batch(output_patches, locations)

    # final output tensor (reconstructed full 3D image)
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

opt = TrainOptions().parse()
epoch = input('Which model epoch do you want to evaluate ?')
title = opt.name

# paths for testing
path = opt.dataroot
path_test = f"./checkpoints/{title}/test_subjects_{title}.txt"


if opt.input_nc==1:
    test_images = load_test_subject_1(path_test,path)
else:
    test_images = load_test_subject_2(path_test,path)
print(test_images)

print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# Change to your chosen saved model
model_path_G = f'./checkpoints/{title}/{epoch}_net_G.pth'
model_path_D = f'./checkpoints/{title}/{epoch}_net_D.pth'


model = load_model(model_path_G,model_path_D, opt)
model.to(device)

# Create the testing set
dataset = Dataset(path, opt.patch_size, opt.input_nc, test_images=test_images, batch_size=1)

ct_paths_check = []
for subject in test_images:
    ct_path_check = subject[0]
    ct_paths_check.append(ct_path_check)

for i in range(len(ct_paths_check)):
    sub = dataset.subjects[i]
    patch_loader, aggregator = dataset.get_subject_loader(sub)

    # Where you want to save the sCT
    output_directory = f'./checkpoints/{title}/Results_{title}'
    run_inference(model, patch_loader, aggregator, output_directory, sub)


# Folder which will be used by BOA (necessary because it requires a specific folder structure)
# destination_folder = '/home/radiology/Documents/Body-and-Organ-Analysis/data'
# organize_images_with_parent(output_directory, destination_folder)

#######################################################################################################################