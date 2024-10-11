# import os
# import torch
# import torchio as tio
# import glob
# from datetime import datetime
# from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
# import numpy as np
#
# def get_subjects(path, transform=None):
#
#     ct_dir = os.path.join(path, 'CT')
#     mri_dir = os.path.join(path, 'MR')
#
#     ct_paths = sorted(glob.glob(ct_dir+'/*.nii.gz'))
#     mri_paths = sorted(glob.glob(mri_dir+'/*.nii.gz'))
#
#
#     assert len(ct_paths) == len(mri_paths)
#
#     subjects = []
#     for (ct_path, mr_path) in (zip(ct_paths, mri_paths)):
#
#         subject = MySubject(
#             ct=tio.ScalarImage(ct_path),
#             mri=tio.ScalarImage(mr_path),
#         )
#         subjects.append(subject)
#     return tio.SubjectsDataset(subjects, transform=transform)
#
#
# def random_split(subjects, ratio=0.9):
#     num_subjects = len(subjects)
#     num_training_subjects = int(ratio * num_subjects)
#     num_test_subjects = num_subjects - num_training_subjects
#
#     num_split_subjects = num_training_subjects, num_test_subjects
#     return torch.utils.data.random_split(subjects, num_split_subjects)
#
# class MySubject(tio.Subject):
#     def check_consistent_attribute(self, *args, **kwargs) -> None:
#         kwargs['relative_tolerance'] = 1e-2
#         kwargs['absolute_tolerance'] = 1e-2
#         return super().check_consistent_attribute(*args, **kwargs)
#
# class Dataset:
#     def __init__(self, path, ratio=0.9, batch_size=1, pretraining=False):
#         self.path = path
#
#         self.batch_size = batch_size
#
#         training_transform = tio.Compose([
#             tio.ToCanonical(),
#             tio.RandomFlip(p=0.5),
#             tio.Clamp(out_min=-1000, out_max=1000, include=['ct']),
#             tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 1000), include=['ct']),
#             tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 1000), exclude=['ct']),
#             tio.RandomAffine(scales=(0.9, 1.2), degrees=15, p=0.5),
#         ])
#
#         validation_transform = tio.Compose([
#             tio.ToCanonical(),
#             tio.Clamp(out_min=-1000, out_max=1000, include=['ct']),
#             tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 1000), include=['ct']),
#             tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 1000), exclude=['ct']),
#         ])
#
#         self.subjects = get_subjects(self.path)
#         self.training_subjects, self.test_subjects = random_split(self.subjects, ratio)
#
#         self.training_transform = training_transform
#         self.test_transform = validation_transform
#
#         self.training_set = tio.SubjectsDataset(
#             self.training_subjects, transform=self.training_transform)
#
#         self.test_set = tio.SubjectsDataset(
#             self.test_subjects, transform=self.test_transform)
#
#     def __len__(self):
#         return len(self.subjects)
#
#     def save_test_set(self):
#         """Save the file paths of the test set subjects."""
#         time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
#         test_subjects_file = f'./checkpoints/test_subjects_{time_now}.txt'
#         with open(test_subjects_file, 'w') as f:
#             for subject in self.test_set:
#                 ct_path = subject['ct'].path
#                 mri_path = subject['mri'].path
#                 f.write(f"CT: {ct_path}, MRI: {mri_path}\n")
#
#         print(f"Test subjects saved to {test_subjects_file}")
#
#     def get_loaders(self):
#         #patch_size = (64, 64, 24)
#         patch_size = (144,144,44)
#         sampler_uniform = tio.data.UniformSampler(patch_size)
#         #samples_per_volume = 40
#         samples_per_volume = 10
#         max_length = 200
#         num_workers = 8
#
#         patches_training_set = tio.Queue(
#             subjects_dataset=self.training_set,
#             max_length=max_length,  # Maximum number of patches that can be stored in the queue
#             samples_per_volume=samples_per_volume,  # Number if patches to be extracted from each volume
#             sampler=sampler_uniform,  # The sampler that we defined previously
#             num_workers=num_workers,  # Number of subprocesses to use for data loading
#             shuffle_subjects=True,
#             shuffle_patches=True,
#         )
#
#         patches_validation_set = tio.Queue(
#             subjects_dataset=self.test_set,
#             max_length=max_length,
#             samples_per_volume=1,
#             sampler=sampler_uniform,
#             num_workers=num_workers,
#             shuffle_subjects=False,
#             shuffle_patches=False,
#         )
#
#
#         training_loader = torch.utils.data.DataLoader(
#             patches_training_set, batch_size=self.batch_size,
#             drop_last=True, shuffle=True)
#
#         test_loader = torch.utils.data.DataLoader(
#             patches_validation_set, batch_size=self.batch_size,
#             drop_last=True, shuffle=False)
#
#         print('Training set:', len(self.training_set), 'subjects')
#         print('Test set:', len(self.test_set), 'subjects')
#         self.save_test_set()
#         return training_loader, test_loader
#
#     def get_val_subject(self, subject):
#         #grid_sampler = tio.inference.GridSampler(subject, patch_size=(64, 64, 24), patch_overlap=8)
#         grid_sampler = tio.inference.GridSampler(subject, patch_size=(144, 144, 44), patch_overlap=16)
#         patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size)
#         aggregator1 = tio.inference.GridAggregator(grid_sampler,'hann')
#         aggregator2 = tio.inference.GridAggregator(grid_sampler,'hann')
#
#         return patch_loader, aggregator1, aggregator2
#
#
#
import os
import torch
import torchio as tio
import glob
from datetime import datetime
import numpy as np


def load_test_subjects(test_file_path):
    """Load test subjects from a text file."""
    test_subjects = set()
    with open(test_file_path, 'r') as f:
        for line in f:
            # Extracting the CT and MRI file paths from the test file
            ct_path, mri_path = line.strip().split(", ")
            ct_path = ct_path.replace("CT: ", "")
            mri_path = mri_path.replace("MRI: ", "")
            test_subjects.add((ct_path, mri_path))
    return test_subjects


def get_subjects(path, test_file_path=None, transform=None):
    ct_dir = os.path.join(path, 'CT')
    mri_dir = os.path.join(path, 'MR')

    ct_paths = sorted(glob.glob(ct_dir + '/*.nii.gz'))
    mri_paths = sorted(glob.glob(mri_dir + '/*.nii.gz'))

    assert len(ct_paths) == len(mri_paths)

    test_subjects = load_test_subjects(test_file_path) if test_file_path else set()

    training_subjects = []
    validation_subjects = []

    for (ct_path, mr_path) in zip(ct_paths, mri_paths):
        subject = MySubject(
            ct=tio.ScalarImage(ct_path),
            mri=tio.ScalarImage(mr_path),
        )
        if (ct_path, mr_path) in test_subjects:
            validation_subjects.append(subject)
        else:
            training_subjects.append(subject)
    print(validation_subjects)
    return training_subjects, validation_subjects


def random_split(subjects, ratio=0.9):
    num_subjects = len(subjects)
    num_training_subjects = int(ratio * num_subjects)
    num_test_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_test_subjects
    return torch.utils.data.random_split(subjects, num_split_subjects)


class MySubject(tio.Subject):
    def check_consistent_attribute(self, *args, **kwargs) -> None:
        kwargs['relative_tolerance'] = 1e-2
        kwargs['absolute_tolerance'] = 1e-2
        return super().check_consistent_attribute(*args, **kwargs)


class Dataset:
    def __init__(self, path, test_file_path=None, ratio=0.9, batch_size=1, pretraining=False):
        self.path = path
        self.batch_size = batch_size

        training_transform = tio.Compose([
            tio.ToCanonical(),
            tio.RandomFlip(p=0.5),
            tio.Clamp(out_min=-1000, out_max=1000, include=['ct']),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 1000), include=['ct']),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 1000), exclude=['ct']),
            tio.RandomAffine(scales=(0.9, 1.2), degrees=15, p=0.5),
        ])

        validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Clamp(out_min=-1000, out_max=1000, include=['ct']),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 1000), include=['ct']),
            tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, 1000), exclude=['ct']),
        ])

        # Load subjects from the dataset folder
        self.training_subjects, self.test_subjects = get_subjects(self.path, test_file_path)

        self.training_transform = training_transform
        self.test_transform = validation_transform

        self.training_set = tio.SubjectsDataset(
            self.training_subjects, transform=self.training_transform)

        self.test_set = tio.SubjectsDataset(
            self.test_subjects, transform=self.test_transform)

    def __len__(self):
        return len(self.training_subjects) + len(self.test_subjects)

    def save_test_set(self):
        """Save the file paths of the test set subjects."""
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        test_subjects_file = f'./checkpoints/test_subjects_{time_now}.txt'
        with open(test_subjects_file, 'w') as f:
            for subject in self.test_set:
                ct_path = subject['ct'].path
                mri_path = subject['mri'].path
                f.write(f"CT: {ct_path}, MRI: {mri_path}\n")

        print(f"Test subjects saved to {test_subjects_file}")

    def get_loaders(self):
        patch_size = (144, 144, 44)
        sampler_uniform = tio.data.UniformSampler(patch_size)
        samples_per_volume = 10
        max_length = 200
        num_workers = 8

        patches_training_set = tio.Queue(
            subjects_dataset=self.training_set,
            max_length=max_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler_uniform,
            num_workers=num_workers,
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

        print('Training set:', len(self.training_set), 'subjects')
        print('Test set:', len(self.test_set), 'subjects')
        self.save_test_set()
        return training_loader, test_loader

    def get_val_subject(self, subject):
        grid_sampler = tio.inference.GridSampler(subject, patch_size=(144, 144, 44), patch_overlap=16)
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size)
        aggregator1 = tio.inference.GridAggregator(grid_sampler,'hann')
        aggregator2 = tio.inference.GridAggregator(grid_sampler,'hann')

        return patch_loader, aggregator1, aggregator2