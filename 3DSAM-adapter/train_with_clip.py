from dataset.datasets import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
import sys
from monai.losses import DiceCELoss, DiceLoss
from modeling.Med_SAM.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
import torch.nn.functional as F
from modeling.Med_SAM.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
import torch
from modeling.Med_SAM.prompt_encoder import PromptEncoder, TwoWayTransformer
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger
from pynvml import *
from modeling.Med_SAM.my_decoder import MyDecoder

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def show_slice(ct, img):
    import cv2
    import numpy as np

    ct = ct[:, 0, :, :, :]
    # 將 `img` 轉換為 [128, 128, 128] 的形狀
    img = np.squeeze(img)
    ct = np.squeeze(ct)

    # 遍歷每一個 slice
    for i in range(img.shape[0]):
        # 獲取當前的 slice
        slice = img[i, :, :]
        ct_slice = ct[i, :, :]

        # 將 slice 正規化到 [0, 255] 的範圍，並轉換為 uint8 類型
        slice = ((slice - slice.min()) * (255.0 / (slice.max() - slice.min()))).astype(np.uint8)
        ct_slice = (
            (ct_slice - ct_slice.min()) * (255.0 / (ct_slice.max() - ct_slice.min()))
        ).astype(np.uint8)

        # 使用 OpenCV 顯示 slice
        cv2.imshow("Slice", slice)
        cv2.imshow("ct_slice", ct_slice)
        cv2.waitKey(0)  # 等待用戶按鍵

    cv2.destroyAllWindows()  # 關閉所有視窗


# python train.py --data colon --snapshot_path "snapshot" --data_prefix "dataset/Task10_Colon"
def main():
    print("init cuda")
    print_gpu_utilization()
    torch.ones((1, 1)).to("cuda")
    print("kernel cuda")
    print_gpu_utilization()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"]
    )
    parser.add_argument("--snapshot_path", default="", type=str)
    parser.add_argument("--data_prefix", default="", type=str)
    parser.add_argument("--rand_crop_size", default=0, nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("-tolerance", default=5, type=int)

    args = parser.parse_args()
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["pancreas", "lits", "colon", "kits"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    train_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker=args.num_worker,
    )
    val_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=False,
        split="val",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker=args.num_worker,
    )
    sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")

    mask_generator = SamAutomaticMaskGenerator(sam)
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,  # 1024
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,  # 16
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice=16,
    )
    state_dict = mask_generator.predictor.model.image_encoder.state_dict()
    img_encoder.load_state_dict(state_dict, strict=False)
    del sam
    img_encoder.to(device)

    print("img_encoder loded cuda")
    print_gpu_utilization()

    # freeze
    for p in img_encoder.parameters():
        p.requires_grad = False

    # depth_embed, slice_embed
    img_encoder.depth_embed.requires_grad = True
    for p in img_encoder.slice_embed.parameters():
        p.requires_grad = True

    # norm1,adapter,norm2
    for i in img_encoder.blocks:
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters():
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
        i.attn.rel_pos_d = nn.parameter.Parameter(
            0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True
        )

    # 3,5,8,11的layer
    for i in img_encoder.neck_3d:
        for p in i.parameters():
            p.requires_grad = True

    # mask_decoder
    mask_decoder = MyDecoder().to(device)
    # mask_decoder = VIT_MLAHead(img_size=96, num_classes=2)
    # mask_decoder.to(device)

    # 3個架構的optimizer和learning rate
    encoder_opt = AdamW(
        [i for i in img_encoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0
    )
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(
        encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500
    )
    # feature_opt = AdamW(parameter_list, lr=args.lr, weight_decay=0)
    # feature_scheduler = torch.optim.lr_scheduler.LinearLR(feature_opt, start_factor=1.0, end_factor=0.01,                                          total_iters=500)
    decoder_opt = AdamW(
        [i for i in mask_decoder.parameters() if i.requires_grad == True],
        lr=args.lr,
        weight_decay=0,
    )
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(
        decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500
    )

    # loss function
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(
        include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5
    )
    best_loss = np.inf
    patch_size = args.rand_crop_size[0]

    # train
    for epoch_num in range(args.max_epoch):
        logging.info(
            f"""
                     ### epoch:{epoch_num} ###
                    """
        )
        loss_summary = []
        img_encoder.train()
        mask_decoder.train()
        print("訓練資料筆數", len(train_data.dataset))
        for idx, (img, seg, spacing) in enumerate(train_data):
            # show_slice(img, seg)
            out = F.interpolate(
                img.float(), scale_factor=512 / patch_size, mode="trilinear"
            )  # 512/128 = 4
            input_batch = out.to(device)
            input_batch = input_batch[0].transpose(0, 1)
            batch_features, feature_list = img_encoder(input_batch)
            # img = [1, 3, 128, 128, 128]
            # seg = [1, 128, 128, 128]
            # out = [1, 3, 512, 512, 512]
            # input_batch = [512, 3, 512, 512]

            feature_list.append(batch_features)
            img_resize = F.interpolate(
                img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device),
                scale_factor=64 / patch_size,
                mode="trilinear",
            )

            print("the features send to decoder")
            for i in feature_list:
                print(i.size())

            # hidden_feature = torch.cat(feature_list,dim=0)
            masks = mask_decoder(feature_list, img_resize)
            masks = masks.permute(0, 1, 4, 2, 3)
            seg = seg.to(device)
            seg = seg.unsqueeze(1)
            loss = loss_cal(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            # feature_opt.zero_grad()
            loss.backward()
            logger.info(
                "epoch: {}/{}, iter: {}/{}".format(epoch_num, args.max_epoch, idx, len(train_data))
                + ": loss:"
                + str(loss_summary[-1].flatten()[0])
            )

            # 設定梯度剪切閾值。 如果更新梯度時梯度超過這個閾值，就會限制在這個範圍內，防止梯度爆炸。
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(prompt_encoder_list[-1].parameters(), 1.0)
            encoder_opt.step()
            # feature_opt.step()
            decoder_opt.step()
        encoder_scheduler.step()
        # feature_scheduler.step()
        decoder_scheduler.step()

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        ## eval模式
        img_encoder.eval()
        mask_decoder.eval()
        ##

        with torch.no_grad():
            loss_summary = []
            for idx, (img, seg, spacing) in enumerate(val_data):
                print("seg: ", seg.sum())
                out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode="trilinear")
                input_batch = out.to(device)
                input_batch = input_batch[0].transpose(0, 1)
                batch_features, feature_list = img_encoder(input_batch)
                feature_list.append(batch_features)
                img_resize = F.interpolate(
                    img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device),
                    scale_factor=64 / patch_size,
                    mode="trilinear",
                )
                # hidden_feature = torch.cat(feature_list,dim=0)
                masks = mask_decoder(feature_list, img_resize)
                masks = masks.permute(0, 1, 4, 2, 3)
                seg = seg.to(device)
                seg = seg.unsqueeze(1)
                loss = dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                logger.info(
                    "epoch: {}/{}, iter: {}/{}".format(
                        epoch_num, args.max_epoch, idx, len(val_data)
                    )
                    + ": loss:"
                    + str(loss_summary[-1].flatten()[0])
                )
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))

        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
        save_checkpoint(
            {
                "epoch": epoch_num + 1,
                "best_val_loss": best_loss,
                "encoder_dict": img_encoder.state_dict(),
                "decoder_dict": mask_decoder.state_dict(),
                # "feature_dict": [i.state_dict() for i in prompt_encoder_list],
                "encoder_opt": encoder_opt.state_dict(),
                # "feature_opt": feature_opt.state_dict(),
                "decoder_opt": decoder_opt.state_dict(),
            },
            is_best=is_best,
            checkpoint=args.snapshot_path,
        )
        logger.info("- Val metrics best: " + str(best_loss))


if __name__ == "__main__":
    main()
