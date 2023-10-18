from dataset.datasets import load_data_volume
import argparse
import numpy as np
import logging
from monai.losses import DiceCELoss, DiceLoss
from modeling.Med_SAM.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.Med_SAM.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from modeling.Med_SAM.prompt_encoder import PromptEncoder, TwoWayTransformer
from functools import partial
import os
from utils.util import setup_logger
import surface_distance
from surface_distance import metrics
import matplotlib.pyplot as plt
from modeling.Med_SAM.my_decoder import MyDecoder
from torch.utils.data import Dataset, DataLoader
from dataset.my_dataset import MyDataset


def plot_slices(img, predict, ground_truth, num_slices, fname, dice, nsd):
    # 把img(有spacing和repeat)內插回來
    img = F.interpolate(img, size=ground_truth.shape[2:], mode="trilinear")
    img = img[0,0]

    predict = predict.squeeze()
    ground_truth = ground_truth.squeeze()

    # different from base_dataset.py
    img = img.cpu().numpy()
    predict = predict.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()

    assert predict.shape == ground_truth.shape, "Image and ground truth must have the same shape"
    assert len(predict.shape) == 3, "Image and ground truth must be 3D tensors"

    # 找出 ground truth 不為 0 的 slices
    non_zero_slices = np.where(np.any(ground_truth != 0, axis=(1, 2)))[0]

    # 如果 non_zero_slices 的數量小於 num_slices，則引發錯誤
    if len(non_zero_slices) < num_slices:
        raise ValueError(
            f"Only {len(non_zero_slices)} non-zero slices found, but {num_slices} were requested."
        )

    # 從 non_zero_slices 中隨機選出 num_slices 個 slices
    slices = np.random.choice(non_zero_slices, num_slices, replace=False)

    fig, axes = plt.subplots(3, num_slices, figsize=(num_slices * 5, 10))

    plt.suptitle(f"DICE: {dice}, NSD: {nsd}")
    for i, slice in enumerate(slices):
        axes[0, i].imshow(img[slice], cmap="gray")
        axes[0, i].set_title("Image")
        axes[1, i].imshow(predict[slice], cmap="gray")
        axes[1, i].set_title("Predict")
        axes[2, i].imshow(ground_truth[slice], cmap="gray")
        axes[2, i].set_title("Ground Truth")
    plt.savefig(f"{fname}.png")
    # plt.show()
    plt.clf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"]
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument(
        "--num_prompts",
        default=1,
        type=int,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument(
        "--checkpoint",
        default="last",
        type=str,
    )
    parser.add_argument("-tolerance", default=5, type=int)
    args = parser.parse_args()
    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))
    # test_data = load_data_volume(
    #     data=args.data,
    #     batch_size=1,
    #     path_prefix=args.data_prefix,
    #     augmentation=False,
    #     split="test",
    #     rand_crop_spatial_size=args.rand_crop_size,
    #     convert_to_sam=False,
    #     do_test_crop=False,
    #     deterministic=True,
    #     num_worker=0,
    # )

    test_data = MyDataset("D:\\ds", "test")
    test_data = DataLoader(test_data, batch_size=1, shuffle=False)
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice=16,
    )
    img_encoder.load_state_dict(
        torch.load(os.path.join(args.snapshot_path, file), map_location="cpu")["encoder_dict"],
        strict=True,
    )
    img_encoder.to(device)

    mask_decoder = MyDecoder().to(device)
    mask_decoder.load_state_dict(
        torch.load(os.path.join(args.snapshot_path, file), map_location="cpu")["decoder_dict"],
        strict=True,
    )
    mask_decoder.to(device)

    dice_loss = DiceLoss(
        include_background=False, softmax=False, to_onehot_y=True, reduction="none"
    )
    img_encoder.eval()
    mask_decoder.eval()

    patch_size = args.rand_crop_size[0]

    def model_predict(img, img_encoder, mask_decoder):
        out = F.interpolate(img.float(), scale_factor=256 / patch_size, mode="trilinear")
        input_batch = out[0].transpose(0, 1)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)
        
        # new_feature => 4個[1,256,32,32,32]tensor的list
        img_resize = F.interpolate( # torch.Size([1, 3, 128, 128, 128])=>torch.Size([1, 1, 64, 64, 64])
            img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device),
            scale_factor=32 / patch_size,
            mode="trilinear",
        )
        
        masks = mask_decoder(feature_list, img_resize)
        masks = masks.permute(0, 1, 4, 2, 3)
        return masks #1,2,128,128,128

    with torch.no_grad():
        loss_summary = []
        loss_nsd = []
        for idx, (img, seg, spacing) in enumerate(test_data):
            seg = seg.float()
            # 把seg (無spacing)內插成img(有spacing)的大小
            prompt = F.interpolate(seg[None, :, :, :, :], img.shape[2:], mode="nearest")[0]
            seg = seg.to(device).unsqueeze(0)
            img = img.to(device)
            
            pred = model_predict(
                img, img_encoder, mask_decoder
            )
            # 把有spacing的結果內插回原始大小
            final_pred = F.interpolate(pred.unsqueeze(1), size=seg.shape[2:], mode="trilinear")
            masks = final_pred > 0.5
            # print("mask_seg", masks.size(),seg.size()) # [1, 1, D, 512, 512], [1, 1, D, 512, 512]
            loss = 1 - dice_loss(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())

            ssd = surface_distance.compute_surface_distances(
                (seg == 1)[0, 0].cpu().numpy(),
                (masks == 1)[0, 0].cpu().numpy(),
                spacing_mm=spacing[0].numpy(),
            )
            nsd = metrics.compute_surface_dice_at_tolerance(ssd, args.tolerance)  # kits
            loss_nsd.append(nsd)
            print(masks.size(),seg.size())
            logger.info(
                " Case {} - Dice {:.6f} | NSD {:.6f}".format(
                    test_data.dataset.img_dict[idx], loss.item(), nsd
                )
            )
        logging.info("- Test metrics Dice: " + str(np.mean(loss_summary)))
        logging.info("- Test metrics NSD: " + str(np.mean(loss_nsd)))


if __name__ == "__main__":
    main()
