# 1. **強度範圍縮放**：這個轉換通常在所有其他轉換之前進行，因為它可以確保影像的強度範圍在後續的轉換中保持一致。

# 2. **隨機偏移強度**：這個轉換可以在強度範圍縮放之後立即進行，因為它是基於影像的原始強度來進行的。

# 3. **裁剪前景**：這個轉換通常在強度相關的轉換之後進行，因為它是基於影像的強度來選擇前景區域的。

# 4. **正規化**：這個轉換通常在所有其他強度相關的轉換之後進行，因為它需要使用影像的全局均值和標準差，這些值可能會被其他強度相關的轉換所改變。

# 5. **資料增強**：如隨機旋轉、隨機縮放和隨機翻轉等，通常在所有前處理步驟之後進行，因為它們不依賴於影像的任何特定屬性。

# 6. **二值化標籤**：這個轉換通常在所有其他轉換之後進行，因為它是最終步驟，不需要再對二值化後的標籤進行任何處理。

# 7. **填充和裁剪**：這些轉換通常在所有其他轉換之後進行，因為它們會改變影像和標籤的空間尺寸。

transforms = [
    # 強度範圍縮放轉換
    ScaleIntensityRanged(
        keys=["image"],
        a_min=intensity_range[0],  # -57
        a_max=intensity_range[1],  # 175
        b_min=intensity_range[0],  # -57
        b_max=intensity_range[1],  # 175
        clip=True,  # 如果縮放後的強度超出了新的範圍，則會被裁剪到該範圍
    ),
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
]

if self.split == "train":  # 強度, crop, norm
    transforms.extend(
        [
            # 正規化
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=global_mean,
                divisor=global_std,
            ),
            RandZoomd(
                keys=["image", "label"],
                prob=0.8,
                min_zoom=0.85,
                max_zoom=1.25,
                mode=["trilinear", "trilinear"],
            ),
            # 將標籤二值化
            BinarizeLabeld(keys=["label"]),
            # 填充到(128*1.2, 128*1.2, 128*1.2)
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=[round(i * 1.2) for i in (128, 128, 128)],
            ),
            # 找到一個cubic，包含至少兩個1和一個0
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                spatial_size=[round(i * 1.2) for i in (128, 128, 128)],
                label_key="label",
                pos=2,
                neg=1,
                num_samples=1,
            ),
            # 裁減回(128,128,128)
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=(128, 128, 128),
                random_size=False,
            ),
            # 在指定的空間軸上隨機翻轉影像和標籤
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # 隨機旋轉影像和標籤 90 度
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ]
    )