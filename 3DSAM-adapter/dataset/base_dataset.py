import pickle
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
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F

import matplotlib.pyplot as plt
def plot_slices(image, ground_truth, num_slices, fname):
    image = image.squeeze()
    ground_truth = ground_truth.squeeze()
    assert image.shape == ground_truth.shape, "Image and ground truth must have the same shape"
    assert len(image.shape) == 3, "Image and ground truth must be 3D tensors"
    
    # 找出 ground truth 不為 0 的 slices
    non_zero_slices = np.where(np.any(ground_truth != 0, axis=(1,2)))[0]
    
    # 如果 non_zero_slices 的數量小於 num_slices，則引發錯誤
    if len(non_zero_slices) < num_slices:
        raise ValueError(f"Only {len(non_zero_slices)} non-zero slices found, but {num_slices} were requested.")
    
    # 從 non_zero_slices 中隨機選出 num_slices 個 slices
    slices = np.random.choice(non_zero_slices, num_slices, replace=False)
    
    fig, axes = plt.subplots(2, num_slices, figsize=(num_slices*5, 10))
    
    for i, slice in enumerate(slices):
        axes[0, i].imshow(image[slice], cmap='gray')
        axes[1, i].imshow(ground_truth[slice], cmap='gray')
        
    plt.savefig(f'{fname}.png')
    plt.show()
    plt.clf()

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


class BaseVolumeDataset(Dataset):
    def __init__(
        self,
        image_paths,
        label_meta,
        augmentation,
        split="train",
        rand_crop_spatial_size=(96, 96, 96),
        convert_to_sam=True,
        do_test_crop=True,
        do_val_crop=True,
        do_nnunet_intensity_aug=True,
    ):
        super().__init__()
        self.img_dict = image_paths
        self.label_dict = label_meta
        self.aug = augmentation
        self.split = split
        self.rand_crop_spatial_size = rand_crop_spatial_size
        self.convert_to_sam = convert_to_sam
        self.do_test_crop = do_test_crop
        self.do_nnunet_intensity_aug = do_nnunet_intensity_aug
        self.do_val_crop = do_val_crop
        self.intensity_range = (
            self.target_spacing
        ) = (
            self.global_mean
        ) = self.global_std = self.spatial_index = self.do_dummy_2D = self.target_class = None

        self._set_dataset_stat()
        self.transforms = self.get_transforms()

    def _set_dataset_stat(self):
        pass

    def __len__(self):
        return len(self.img_dict)  # ['path1','path2']

    def __getitem__(self, idx):
        img_path = self.img_dict[idx]
        label_path = self.label_dict[idx]
        img_vol = nib.load(img_path)
        # 獲取影像數據，並將其轉換為numpy陣列
        print("原始影像大小", img_vol.shape)  # ex: (512, 512, 178)
        img = (
            img_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        )  # transposr((2,1,0))
        # print(img.shape) # ex: (178, 512, 512)
        # 找出所有的唯一值

        # img_vol.header.get_zooms()用來獲取影像的空間解析度
        img_spacing = tuple(np.array(img_vol.header.get_zooms())[self.spatial_index])
        # print(img_spacing) # ex: (3.0, 0.677794, 0.677734), spacing 越大解析度越差

        seg_vol = nib.load(label_path)
        seg = seg_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)

        img[np.isnan(img)] = 0
        seg[np.isnan(seg)] = 0
        seg = (seg == self.target_class).astype(np.float32)

        # 影像的最大解析度與最小解析度之間的比例是否大於8
        if (np.max(img_spacing) / np.min(img_spacing) > 8) or (
            np.max(self.target_spacing / np.min(self.target_spacing) > 8)
        ):
            print("解析度差8倍")
            # resize 2D
            img_tensor = F.interpolate(
                input=torch.tensor(img[:, None, :, :]),
                scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                mode="bilinear",
            )

            if self.split != "test":
                seg_tensor = F.interpolate(
                    input=torch.tensor(seg[:, None, :, :]),
                    scale_factor=tuple(
                        [img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]
                    ),
                    mode="bilinear",
                )
            img = (
                F.interpolate(
                    input=img_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                    scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                    mode="trilinear",
                )
                .squeeze(0)
                .numpy()
            )

            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=seg_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )
        else:
            img = (
                # 想在解析度上內插回一個正方體，但影像張量大小會改變
                F.interpolate(
                    input=torch.tensor(img[None, None, :, :, :]),
                    scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(3)]),
                    mode="trilinear",
                )
                .squeeze(0)
                .numpy()
            )
            # 不是testing的話label也要一同spacing (seg 結果不是[0, 1])
            if self.split != "test":
                seg = (
                    F.interpolate(
                        input=torch.tensor(seg[None, None, :, :, :]),
                        scale_factor=tuple(
                            [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                        ),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )

        print("原始影像做完spacing後的大小", img.shape)
        print("mask做完spacing後的大小", seg.shape)

        # plot_slices(img, seg, 5, "after_spacing")
        if (self.aug and self.split == "train") or ((self.do_val_crop and self.split == "val")):
            trans_dict = self.transforms({"image": img, "label": seg})[0]
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        else:
            trans_dict = self.transforms({"image": img, "label": seg})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        # plot_slices(img_aug, seg_aug, 5, "after_aug")
        seg_aug = seg_aug.squeeze()  # 移除所有1維度
        print(seg_aug.size()) # [1,128,128,128]
        img_aug = img_aug.repeat(3, 1, 1, 1)  # 複製3遍
        print(img_aug.size()) # [3,128,128,128]


        return img_aug, seg_aug, np.array(img_vol.header.get_zooms())[self.spatial_index]

    def get_transforms(self):
        transforms = [
            # 強度範圍縮放轉換
            # a_min, a_max => b_min, b_max (這裡實際上沒差)
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.intensity_range[0], # -57
                a_max=self.intensity_range[1], # 175
                b_min=self.intensity_range[0], # -57
                b_max=self.intensity_range[1], # 175
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
                        select_fn=lambda x: x > self.intensity_range[0],  # 根據圖像的強度範圍來選擇前景區域
                    ),
                    # 正規化
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                ]
            )

            if self.do_dummy_2D: # 旋轉縮放
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
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                    ),
                    # 找到一個cubic，包含至少兩個1和一個0
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                        label_key="label",
                        pos=2,
                        neg=1,
                        num_samples=1,
                    ),
                    # 裁減回(128,128,128)
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
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
        elif (not self.do_val_crop) and (self.split == "val"):
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif (self.do_val_crop) and (self.split == "val"):
            transforms.extend(
                [
                    # CropForegroundd(
                    #     keys=["image", "label"],
                    #     source_key="image",
                    #     select_fn=lambda x: x > self.intensity_range[0],
                    # ),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[i for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif self.split == "test":
            transforms.extend(
                [
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError

        transforms = Compose(transforms)

        return transforms
