import pickle
import os
import json
import glob
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

target_spacing = (1, 1, 1)

def MSD_walk(base_path=None):
    data_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(data_dirs)
    image_paths = []
    label_paths = []
    for dir in data_dirs:
        if dir in ["Task03_Liver", "Task06_Lung", "Task07_Pancreas", "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon"]:
            image_path = os.path.join(base_path, dir, "imagesTr")
            label_path = os.path.join(base_path, dir, "labelsTr")
            a = glob.glob(os.path.join(image_path, "*.nii.gz"))
            b = glob.glob(os.path.join(label_path, "*.nii.gz"))
            assert len(a) == len(b)
            image_paths.append(a)
            label_paths.append(b)

    return image_paths, label_paths


def save_images(image_pathlist, label_pathlist, name):
    # split train, val, test 8:1:1
    train_images, val_test_images, train_labels, val_test_labels = train_test_split(image_pathlist, label_pathlist, test_size=0.2, random_state=42)
    val_images, test_images, val_labels, test_labels = train_test_split(val_test_images, val_test_labels, test_size=0.5, random_state=42)
    # loop 3 set
    for image_pathlist, label_pathlist, split in [(train_images, train_labels, "train"), (val_images, val_labels, "val"), (test_images, test_labels, "test")]:
        # loop each image, label
        for img_path, label_path in zip(image_pathlist, label_pathlist):
            img_vol = nib.load(img_path)
            img = img_vol.get_fdata().astype(np.float32).transpose(2, 1, 0)
            img_spacing = tuple(np.array(img_vol.header.get_zooms())[[2, 1, 0]])
            seg_vol = nib.load(label_path)
            seg = seg_vol.get_fdata().astype(np.float32).transpose(2, 1, 0)
            # print(img_spacing)
            # print(img.shape)
            # print(seg.shape)

            img[np.isnan(img)] = 0
            seg[np.isnan(seg)] = 0

            # 影像的最大解析度與最小解析度之間的比例是否大於8
            if np.max(img_spacing) / np.min(img_spacing) > 8:
                # resize 2D
                img_tensor = F.interpolate(
                    input=torch.tensor(img[:, None, :, :]),
                    scale_factor=tuple([img_spacing[i] / target_spacing[i] for i in range(1, 3)]),
                    mode="bilinear",
                )

                if split !="val" and split != "test":
                    seg_tensor = F.interpolate(
                        input=torch.tensor(seg[:, None, :, :]),
                        scale_factor=tuple([img_spacing[i] / target_spacing[i] for i in range(1, 3)]),
                        mode="bilinear",
                    )
                img = (
                    F.interpolate(
                        input=img_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                        scale_factor=(img_spacing[0] / target_spacing[0], 1, 1),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )

                if split !="val" and split != "test":
                    seg = (
                        F.interpolate(
                            input=seg_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                            scale_factor=(img_spacing[0] / target_spacing[0], 1, 1),
                            mode="trilinear",
                        )
                        .squeeze(0)
                        .numpy()
                    )
            else:
                # resize 3D
                img = (
                    # 想在解析度上內插回一個正方體，但影像張量大小會改變
                    F.interpolate(
                        input=torch.tensor(img[None, None, :, :, :]),
                        scale_factor=tuple([img_spacing[i] / target_spacing[i] for i in range(3)]),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )
                # 不是testing的話label也要一同spacing (seg 結果不是[0, 1])
                if split !="val" and split != "test":
                    seg = (
                        F.interpolate(
                            input=torch.tensor(seg[None, None, :, :, :]),
                            scale_factor=tuple([img_spacing[i] / target_spacing[i] for i in range(3)]),
                            mode="trilinear",
                        )
                        .squeeze(0)
                        .numpy()
                    )
            print(img.shape)
            print(seg.shape)
            if not os.path.isdir(f"E:\\ds\\{name}\\{split}\\image"):
                os.makedirs(f"E:\\ds\\{name}\\{split}\\image", mode=0o777)
            if not os.path.isdir(f"E:\\ds\\{name}\\{split}\\label"):
                os.makedirs(f"E:\\ds\\{name}\\{split}\\label", mode=0o777)
            np.save(f"E:\\ds\\{name}\\{split}\\image\\{img_path.split('/')[-1]}", img)
            np.save(f"E:\\ds\\{name}\\{split}\\label\\{label_path.split('/')[-1]}", seg)


if __name__ == "__main__":
    image_paths, label_paths = MSD_walk(base_path="E:\\SAM")
    for image_pathlist, label_pathlist, name in zip(image_paths, label_paths, ["Task03_Liver", "Task06_Lung", "Task07_Pancreas", "Task08_HepaticVessel", "Task09_Spleen", "Task10_Colon"]):
        save_images(image_pathlist, label_pathlist, name)
