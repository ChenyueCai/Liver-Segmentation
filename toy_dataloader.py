# This toy data loader takes in a DICOM file and a NIFIT
# Convert them to numpy array

import pydicom
import nibabel as nib
from PIL import Image
import skimage.io as skio
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


# Convert a nii file to a numpy array with size (num_slice, h, w)
# Convert a directory of dcm files to a numpy array with size (num_files, h, w)
# Note: dcm files are sorted in descending order to fit the slice sequence of nii file
def convert_to_numpy(filename=None, directory=None, file_type=None):
    if file_type == 'nii':
        assert filename.endswith('.nii.gz')
        nii_array = np.array(nib.load(filename).get_fdata())
        h, w, num_slice = nii_array.shape
        seg_nii = np.zeros((num_slice, h, w))
        for i in range(num_slice):
            seg_nii[i, :, :] = nii_array[:, :, i]
        return seg_nii
    elif file_type == 'dcm dir':
        dcm_directory_contents = sorted(os.listdir(directory), reverse=True)
        img_dcm = []
        for dcm_file in dcm_directory_contents:
            if dcm_file.endswith('.dcm'):
                img_dcm.append(pydicom.dcmread(directory + '/' + dcm_file).pixel_array)
        img_dcm = np.array(img_dcm)
        return img_dcm
    else:
        return None


def convert_to_3channels(img, channel_space):
    n, h, w = img.shape
    img_3c = np.zeros((n,h,w,3))
    for i, _ in enumerate(img_3c):
        img_3c[i,:,:,0] = img[np.max([i-channel_space, 0])]
        img_3c[i,:,:,1] = img[i]
        img_3c[i,:,:,2] = img[np.min([i+channel_space, len(img)-1])]
    return img_3c


class LiverSegSet(Dataset):

    def __init__(self, ct_dir, seg_dir, h, w):
        """
        Args:
            ct_dir: a directory with all patients ct scan results (in directory form, dcm files within)
            seg_dir: a directory with all nii segmentation files within
        """
        self.h = h
        self.w = w

        dcm_dirs = sorted(os.listdir(ct_dir))  # Assume correspondence between ct scan and seg file names
        nii_files = sorted(os.listdir(seg_dir))
        self.ct_image = []
        self.seg_mask = []
        for dcm_dir in dcm_dirs:
            if os.path.isdir(ct_dir + '/' + dcm_dir):
                self.ct_image.append(convert_to_3channels(
                    convert_to_numpy(directory=ct_dir + '/' + dcm_dir, file_type='dcm dir'), 3))
        self.ct_image = np.array(self.ct_image).reshape((-1, self.h, self.w, 3))
        for nii_file in nii_files:
            if nii_file.endswith('.nii.gz'):
                self.seg_mask.append(
                    convert_to_3channels(convert_to_numpy(filename=seg_dir + '/' + nii_file, file_type='nii'), 3))
        self.seg_mask = np.array(self.seg_mask).reshape((-1, self.h, self.w, 3))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.ct_image[idx].astype(np.float32), self.seg_mask[idx].astype(np.float32)


def main():

    data_set = LiverSegSet(ct_dir='ct_dirs', seg_dir='seg_files', h=512, w=512)
    for i in range(4):
        skio.imsave('dataset_ex/ct'+ str(i) + '.jpg', data_set[i][0])
        skio.imsave('dataset_ex/seg' + str(i)+'.jpg', data_set[i][1])



if __name__ == '__main__':
    main()
