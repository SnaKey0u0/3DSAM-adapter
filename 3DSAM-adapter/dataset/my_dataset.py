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

class BinarizeLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        threshold: float = 0.5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])

            dtype = d[key].dtype
            d[key] = (d[key] > self.threshold).to(dtype)
        return d

class MyDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.root_dir = root_dir
        self.image_paths = []
        self.label_paths = []
        self.dataset_names = []
        self.split = mode
        self.transforms_kits = self.get_transforms(
            do_dummy_2D=False, global_mean=59.53867, global_std=55.457336, intensity_range=(-54, 247)
        )
        self.transforms_lits = self.get_transforms(
            do_dummy_2D=False, global_mean=60.057533, global_std=40.198017, intensity_range=(-48, 163)
        )
        self.transforms_pancreas = self.get_transforms(
            do_dummy_2D=True, global_mean=68.45214, global_std=63.422806, intensity_range=(-39, 204)
        )
        self.transforms_colon = self.get_transforms(
            do_dummy_2D=True, global_mean=65.175035, global_std=32.651197, intensity_range=(-57, 175)
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

        # 讀取.npy檔案
        image = np.load(image_path)
        label = np.load(label_path)

        # 將numpy array轉換為torch tensor
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label)

        # return image_tensor, label_tensor, dataset_name
        
        if self.split == "train" or self.split == "val":
            if dataset_name == "colon":
                trans_dict = self.transforms_colon({"image": image_tensor, "label": label_tensor})[0]
                img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
            elif dataset_name == "kits":
                trans_dict = self.transforms_kits({"image": image_tensor, "label": label_tensor})[0]
                img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
            elif dataset_name == "lits":
                trans_dict = self.transforms_lits({"image": image_tensor, "label": label_tensor})[0]
                img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
            elif dataset_name == "pancreas":
                trans_dict = self.transforms_pancreas({"image": image_tensor, "label": label_tensor})[0]
                img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        else:
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

    def get_transforms(self, do_dummy_2D=True, global_mean=None, global_std=None, intensity_range=None):
        transforms = [
            # 強度範圍縮放轉換
            # a_min, a_max => b_min, b_max (這裡實際上沒差)
            ScaleIntensityRanged(
                keys=["image"],
                a_min=intensity_range[0], # -57
                a_max=intensity_range[1], # 175
                b_min=intensity_range[0], # -57
                b_max=intensity_range[1], # 175
                clip=True,  # 如果縮放後的強度超出了新的範圍，則會被裁剪到該範圍
            ),
        ]

        if self.split == "train": # 強度, crop, norm
            transforms.extend(
                [
                    # 以0.5的機率，對"image"的強度進行最多±20的隨機偏移
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=20,
                        prob=0.5,
                    ),
                    # 根據source_key="image"的圖像來裁剪前景區域，
                    # 並將裁剪後的圖像和標籤存放在keys=[“image”, “label”]中
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                        select_fn=lambda x: x > intensity_range[0],  # 根據圖像的強度範圍來選擇前景區域
                    ),
                    # 正規化
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=global_mean,
                        divisor=global_std,
                    ),
                ]
            )

            if do_dummy_2D: # 旋轉縮放
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
                        # RandRotated(
                        #     keys=["image", "label"],
                        #     prob=0.3,
                        #     range_x=30 / 180 * np.pi,
                        #     range_y=30 / 180 * np.pi,
                        #     range_z=30 / 180 * np.pi,
                        #     keep_size=False,
                        # ),
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.8,
                            min_zoom=0.85,
                            max_zoom=1.25,
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )

            transforms.extend(
                [
                    # 將標籤二值化
                    BinarizeLabeld(keys=["label"]),
                    # 填充到(128*1.2, 128*1.2, 128*1.2)
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in (128,128,128)],
                    ),
                    # 找到一個cubic，包含至少兩個1和一個0
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in (128,128,128)],
                        label_key="label",
                        pos=2,
                        neg=1,
                        num_samples=1,
                    ),
                    # 裁減回(128,128,128)
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=(128,128,128),
                        random_size=False,
                    ),
                    # 在指定的空間軸上隨機翻轉影像和標籤
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    # 隨機旋轉影像和標籤 90 度
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                    # RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
                    # RandShiftIntensityd(
                    #     keys=["image"],
                    #     offsets=0.10,
                    #     prob=0.2,
                    # ),
                    # RandGaussianNoised(keys=["image"], prob=0.1),
                    # RandGaussianSmoothd(
                    #     keys=["image"],
                    #     prob=0.2,
                    #     sigma_x=(0.5, 1),
                    #     sigma_y=(0.5, 1),
                    #     sigma_z=(0.5, 1),
                    # ),
                    # AddChanneld(keys=["image", "label"]),
                    # RandShiftIntensityd(keys=["image"], offsets=10),
                    # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),]
                ]
            )
            # end of train transform
        # elif (not self.do_val_crop) and (self.split == "val"):
        #     transforms.extend(
        #         [
        #             CropForegroundd(
        #                 keys=["image", "label"],
        #                 source_key="image",
        #             ),
        #             BinarizeLabeld(keys=["label"]),
        #         ]
        #     )
        elif self.split == "val":
            transforms.extend(
                [
                    # CropForegroundd(
                    #     keys=["image", "label"],
                    #     source_key="image",
                    #     select_fn=lambda x: x > self.intensity_range[0],
                    # ),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[i for i in (128,128,128)],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=(128,128,128),
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=global_mean,
                        divisor=global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif self.split == "test":
            transforms.extend(
                [
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=global_mean,
                        divisor=global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError

        transforms = Compose(transforms)

        return transforms


if __name__ == "__main__":
    dataset = MyDataset("D:\\ds", "train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for images, labels, dataset_names in dataloader:
        print(images.size())
        print(labels.size(), labels.max())
        print(dataset_names)
