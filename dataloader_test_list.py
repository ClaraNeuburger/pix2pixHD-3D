import os
import torch
import torchio as tio
import glob
from datetime import datetime
import numpy as np
import re
import math

# Added code to correct the issue with ToCanonical: the origins of the CT and MRI are slightly different even
# when using the correct transforms
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

# Loading the subjects from the given paths, apply the transforms for pre processing
def get_subjects(path, nb_channels, plot_data, test_file_path, transform=None):

    ### If the MRI image only has one channel, simply load the 3 images
    if nb_channels == 1:
        ct_dir = os.path.join(path, 'CT')
        mri_dir = os.path.join(path, 'MR')

        ct_paths = sorted(glob.glob(ct_dir + '/*.nii.gz'))
        mri_paths = sorted(glob.glob(mri_dir + '/*.nii.gz'))

        assert len(ct_paths) == len(mri_paths)

        #test_subjects = load_test_subject_1(test_file_path) if test_file_path else set()
        test_subjects = load_test_subject_1(test_file_path,path)
        training_subjects = []
        validation_subjects = []
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
                validation_subjects.append(subject)
            else:

                training_subjects.append(subject)


    ### If the MRI image has 2 channels aka if you use the fat and water sequence from the dixon
    if nb_channels == 2:
        ct_dir = os.path.join(path, 'CT')
        mri_W_dir = os.path.join(path, 'MR water')
        mri_F_dir = os.path.join(path, 'MR fat resized')

        ct_paths = sorted(glob.glob(ct_dir + '/*.nii.gz'))
        mri_W_paths = sorted(glob.glob(mri_W_dir + '/*.nii.gz'))
        mri_F_paths = sorted(glob.glob(mri_F_dir + '/*.nii.gz'))
        assert len(ct_paths) == len(mri_W_paths)

        # test_subjects = load_test_subject_2(test_file_path) if test_file_path else set()
        test_subjects = load_test_subject_2(test_file_path, path)


        training_subjects = []
        validation_subjects = []
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
                validation_subjects.append(subject)
            else:
                training_subjects.append(subject)
        if plot_data:
            training_subjects[0].plot()

    return training_subjects,validation_subjects


def random_split(subjects, ratio=0.9):
    num_subjects = len(subjects)
    num_training_subjects = int(ratio * num_subjects)
    num_test_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_test_subjects
    return torch.utils.data.random_split(subjects, num_split_subjects)


class Dataset_testlist:
    def __init__(self, path, nb_channels, plot_data, title, testing_dataroot,patch_size,patch_overlap,ratio=0.9, batch_size=1, pretraining=False):
        self.path = path
        self.nb_channels = nb_channels
        self.plot_data = plot_data
        self.title = title
        self.batch_size = batch_size
        self.test_file_path = testing_dataroot
        self.patch_size = patch_size
        self.patch_overlap= patch_overlap
        if nb_channels == 1:
            training_transform = tio.Compose([
                tio.ToCanonical(),
                tio.RandomFlip(p=0.5),
                tio.Clamp(out_min=-1000, out_max=1000, include=['ct']),
                tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 1000), include=['ct']),
                tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 1000), include=['mri']),
                tio.RandomAffine(scales=(0.9, 1.2), degrees=15, p=0.5),
            ])

            validation_transform = tio.Compose([
                tio.ToCanonical(),
                tio.Clamp(out_min=-1000, out_max=1000, include=['ct']),
                tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 1000), include=['ct']),
                tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 1000), include=['mri']),
            ])

        if nb_channels == 2:
            training_transform = tio.Compose([
                tio.ToCanonical(),
                tio.RandomFlip(p=0.5),
                tio.Clamp(out_min=-1000, out_max=1000, include=['ct']),
                tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 1000), include=['ct']),
                tio.RandomAffine(scales=(0.9, 1.2), degrees=15, p=0.5),
            ])

            validation_transform = tio.Compose([
                tio.ToCanonical(),
                tio.Clamp(out_min=-1000, out_max=1000, include=['ct']),
                tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 1000), include=['ct']),
            ])

        self.training_subjects, self.test_subjects = get_subjects(self.path,self.nb_channels, self.plot_data,test_file_path=self.test_file_path)

        self.training_transform = training_transform
        self.test_transform = validation_transform

        self.training_set = tio.SubjectsDataset(
            self.training_subjects, transform=self.training_transform)

        self.test_set = tio.SubjectsDataset(
            self.test_subjects, transform=self.test_transform)

    def __len__(self):
        return len(self.subjects)

    def save_test_set(self):
        test_subjects_file = f"./checkpoints/{self.title}/test_subjects_{self.title}.txt"
        # test_subjects_file = f'./checkpoints/test_subjects/test_subjects_{self.title}.txt'

        with open(test_subjects_file, 'w') as f:
            for subject in self.test_set:
                ct_path = subject['ct'].path

                last_segment = ct_path.name

                parts = last_segment.split('_')

                patient_nb = parts[0]
                scan_nb_all = parts[2]
                parts_end = scan_nb_all.split('.')
                scan_nb = parts_end[0]

                f.write(f"{patient_nb}_{scan_nb}\n")

        print(f"Test subjects saved to {test_subjects_file}")

    def get_loaders(self):
        sampler_uniform = tio.data.UniformSampler(self.patch_size)

        """
        replace with a computation to automatically get the good number of samples required for the image dimension 
        based on the chosen path size 
        width/patch_size[0] rounded * depth/patch_size[2] rounded 
        """
        width = self.training_subjects[0].ct.shape[1]
        depth = self.training_subjects[0].ct.shape[3]

        samples_per_volume = math.ceil(width/self.patch_size[0])*math.ceil(depth/self.patch_size[2])

        max_length = 200
        num_workers = 0

        patches_training_set = tio.Queue(
            subjects_dataset=self.training_set,
            max_length=max_length,  # Maximum number of patches that can be stored in the queue
            samples_per_volume=samples_per_volume,  # Number if patches to be extracted from each volume
            sampler=sampler_uniform,  # The sampler that we defined previously
            num_workers=num_workers,  # Number of subprocesses to use for data loading
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        patches_validation_set = tio.Queue(
            subjects_dataset=self.test_set,
            max_length=max_length,
            samples_per_volume=1,
            sampler=sampler_uniform,
            num_workers=num_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
        )


        training_loader = torch.utils.data.DataLoader(
            patches_training_set, batch_size=self.batch_size,
            drop_last=True, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            patches_validation_set, batch_size=self.batch_size,
            drop_last=True, shuffle=False)

        self.save_test_set()
        print('Training set:', len(self.training_set), 'subjects')
        print('Test set:', len(self.test_set), 'subjects')
        return training_loader, test_loader

    def get_val_subject(self, subject):
        grid_sampler = tio.inference.GridSampler(subject, patch_size=self.patch_size, patch_overlap=self.patch_overlap)
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size)
        aggregator1 = tio.inference.GridAggregator(grid_sampler,'hann')
        aggregator2 = tio.inference.GridAggregator(grid_sampler,'hann')

        return patch_loader, aggregator1, aggregator2



