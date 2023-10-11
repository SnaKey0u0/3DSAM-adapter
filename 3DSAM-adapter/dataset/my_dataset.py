import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms import (
    Compose,
    AddChanneld,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandAffined,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
)


class MyDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.root_dir = root_dir
        self.image_paths = []
        self.label_paths = []
        self.dataset_names = []
        self.split = mode
        self.transforms_kits = self.get_transforms(
            do_dummy_2D=False, global_mean=59.53867, global_std=55.457336
        )
        self.transforms_lits = self.get_transforms(
            do_dummy_2D=False, global_mean=60.057533, global_std=40.198017
        )
        self.transforms_pancreas = self.get_transforms(
            do_dummy_2D=True, global_mean=68.45214, global_std=63.422806
        )
        self.transforms_colon = self.get_transforms(
            do_dummy_2D=True, global_mean=65.175035, global_std=32.651197
        )

        # 遍歷所有子資料夾
        for dataset in os.listdir(root_dir):
            train_dir = os.path.join(root_dir, dataset, mode)
            image_dir = os.path.join(train_dir, "image")
            label_dir = os.path.join(train_dir, "label")

            # 確保資料夾存在
            if os.path.exists(image_dir) and os.path.exists(label_dir):
                # 獲取所有.npy檔案的路徑
                for file in os.listdir(image_dir):
                    if file.endswith(".npy"):
                        self.image_paths.append(os.path.join(image_dir, file))
                        self.label_paths.append(os.path.join(label_dir, file))
                        self.dataset_names.append(dataset)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        dataset_name = self.dataset_names[idx]

        # 讀取.npy檔案並轉換為torch.Tensor
        image = np.load(image_path)
        label = np.load(label_path)

        # 將numpy array轉換為torch tensor
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label).unsqueeze(0)

        # return image_tensor, label_tensor, dataset_name
        
        if dataset_name == "colon":
            trans_dict = self.transforms_colon({"image": image_tensor, "label": label_tensor})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        elif dataset_name == "kits":
            trans_dict = self.transforms_kits({"image": image_tensor, "label": label_tensor})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        elif dataset_name == "lits":
            trans_dict = self.transforms_lits({"image": image_tensor, "label": label_tensor})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        elif dataset_name == "pancreas":
            trans_dict = self.transforms_pancreas({"image": image_tensor, "label": label_tensor})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]

        seg_aug = seg_aug.squeeze()  # 移除所有1維度
        img_aug = img_aug.repeat(3, 1, 1, 1)  # 複製3遍
        return img_aug, seg_aug, dataset_name

    def get_transforms(self, do_dummy_2D=True, global_mean=None, global_std=None):
        transforms = []
        if self.split == "train":  # 強度, crop, norm
            transforms.extend(
                [
                    # 以0.5的機率，對"image"的強度進行最多±20的隨機偏移
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=20,
                        prob=0.5,
                    ),
                    # 正規化
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=global_mean,
                        divisor=global_std,
                    ),
                ]
            )

            if do_dummy_2D:  # 旋轉縮放
                transforms.extend(
                    [
                        # 隨機旋轉，旋轉的角度範圍為±30度
                        RandRotated(
                            keys=["image", "label"],
                            prob=0.3,
                            range_x=30 / 180 * np.pi,
                            keep_size=False,
                        ),
                        # 隨機縮放到原始大小的90%到110%之間。但影像大小不變
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.3,
                            min_zoom=[1, 0.9, 0.9],
                            max_zoom=[1, 1.1, 1.1],
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )
            else:
                transforms.extend(
                    [
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.8,
                            min_zoom = 0.85,
                            max_zoom = 1.25,
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )

            transforms.extend(
                [
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=(128,128,128),
                        random_size=False,
                    ),
                    # 在指定的空間軸上隨機翻轉影像和標籤
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    # # 隨機旋轉影像和標籤 90 度
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                ]
            )
            # end of train transform

        # val和test都處理完才存成npy了，load後不用再做

        transforms = Compose(transforms)

        return transforms


if __name__ == "__main__":
    dataset = MyDataset("D:\\ds", "train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for images, labels, dataset_names in dataloader:
        print(images.size())
        print(labels.size(), labels.max())
        print(dataset_names)
