from segmentation.toy_dataloader import *


def patient_liver_id(seg_masks):
    liver_id = []
    for s_m in seg_masks:
        liver_id.append(has_liver(s_m))
    return np.array(liver_id)


def has_liver(seg_mask):
    seg_mask = np.array(seg_mask)
    if np.amax(seg_mask) == 0 and np.amin(seg_mask) == 0:
        return 0
    else:
        return 1


class LiverIDSet(Dataset):

    def __init__(self, ct_dir, seg_dir, h, w):
        """
        Args:
            ct_dir: a directory with all patients ct scan results (in directory form, dcm files within)
            seg_dir: a directory with all nii segmentation files within
            h: height of the image
            w: width of the image
        """
        self.h = h
        self.w = w

        dcm_dirs = sorted(os.listdir(ct_dir))  # Assume correspondence between ct scan and seg file names
        nii_files = sorted(os.listdir(seg_dir))
        self.ct_image = []
        self.organ_id = []
        self.seg_mask = []
        for dcm_dir in dcm_dirs:
            if os.path.isdir(ct_dir + '/' + dcm_dir):
                self.ct_image.append(convert_to_3channels(
                    convert_to_numpy(directory=ct_dir + '/' + dcm_dir, file_type='dcm dir'), 3))
        ct_images = self.ct_image
        self.ct_image = self.ct_image[0]
        for i in range(1, len(ct_images)):
            self.ct_image = np.vstack((self.ct_image, ct_images[i]))
        self.ct_image = np.array(self.ct_image).reshape((-1, 3, self.h, self.w))

        for nii_file in nii_files:
            if nii_file.endswith('.nii.gz'):
                self.seg_mask.append(
                    convert_to_numpy(filename=seg_dir + '/' + nii_file, file_type='nii'))
        seg_masks = self.seg_mask
        self.seg_mask = self.seg_mask[0]
        for i in range(1, len(seg_masks)):
            self.seg_mask = np.vstack((self.seg_mask, seg_masks[i]))
        self.seg_mask = np.array(self.seg_mask).reshape((-1, self.h, self.w)) # Edited for the dataloader
        self.organ_id = patient_liver_id(self.seg_mask)

    def __len__(self):
        return len(self.ct_image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.ct_image[idx].astype(np.float32), self.organ_id[idx].astype(np.float32)