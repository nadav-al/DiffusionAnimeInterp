import torch.utils.data as data
from PIL import Image
import os

def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # cv2.imwrite(resize)
        # Crop image if crop area specified.
        if cropArea != None:
            cropped_img = resized_img.crop(cropArea)
        else:
            cropped_img = resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img


        return flipped_img.convert('RGB')


class AniPairs(data.Dataset):
    """
    A dataset class representing pairs of images (clean, distorted) from a folder structure.
    """

    def __init__(self, root_dir, transforms=None):
        """
        Args:
          root_dir (str): Path to the root directory containing image pairs.
          transforms (callable, optional): Function to apply transformations to images.
        """
        self.root_dir = root_dir
        self.image_pairs = []
        self.transforms = transforms

        # Find all subdirectories containing clean and distorted images
        for subdir in os.listdir(root_dir):
            subpath = os.path.join(root_dir, subdir)
            if os.path.isdir(subpath):
                clean_path = os.path.join(subpath, "clean_1.png")
                distorted_path = os.path.join(subpath, "distorted_1.png")
                if os.path.isfile(clean_path) and os.path.isfile(distorted_path):
                    self.image_pairs.append((clean_path, distorted_path))

    def __len__(self):
        """
        Returns the number of image pairs in the dataset.

        Returns:
          int: Length of the dataset.
        """
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Loads a pair of images (clean, distorted) at the given index.

        Args:
          idx (int): Index of the image pair to access.

        Returns:
          tuple: Tuple containing the loaded clean and distorted images.
        """
        clean_path, distorted_path = self.image_pairs[idx]
        clean_image = _pil_loader(clean_path)
        distorted_image = _pil_loader(distorted_path)

        if self.transforms:
            clean_image = self.transforms(clean_image)
            distorted_image = self.transforms(distorted_image)

        return clean_image, distorted_image
