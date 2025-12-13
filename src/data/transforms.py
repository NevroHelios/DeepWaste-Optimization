from torchvision import transforms
from torchvision.transforms import Compose
from albumentations import (
    Resize, HorizontalFlip, VerticalFlip, RandomRotate90,
    ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue,
    RandomResizedCrop, CoarseDropout, Normalize
)
import cv2
from typing import Literal, Optional, Tuple

class Transform:
    def __init__(self, img_size):
        self.img_size = img_size
    
    def get_transforms(self, mode: Optional[Literal['augment']],
                       aug_strength:float=1.0
            ) -> Tuple[Compose, Compose]:
        """returns train & val transforms"""
        shift_limit = 0.02 * aug_strength
        scale_limit = 0.1 * aug_strength
        rotate_limit = int(10 * aug_strength)
        hue_shift = int(10 * aug_strength)
        sat_shift = int(10 * aug_strength)
        val_shift = int(10 * aug_strength)
        brightness_limit = 0.15 * aug_strength
        contrast_limit = 0.15 * aug_strength
        dropout_p = min(0.3 * aug_strength, 1.0)

        train_transform = None
        val_transform = Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

        if mode == 'augment':
            train_transform = Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=0.3),
                Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                transforms.ToTensor(),
            ])

        return (train_transform, val_transform) if train_transform else (val_transform, val_transform)
    