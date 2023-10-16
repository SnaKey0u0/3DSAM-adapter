import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from torch.utils.checkpoint import checkpoint

from segment_anything.modeling.common import LayerNorm2d, MLPBlock
from segment_anything.modeling.image_encoder import (
    Attention,
    PatchEmbed,
    window_partition,
    window_unpartition,
)
import logging
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)

# logging.basicConfig(
#     filename="my.log",
#     filemode="w+",
#     encoding="utf-8",
#     format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
#     datefmt="%H:%M:%S",
#     level=logging.INFO,
# )


class Adapter(nn.Module):
    def __init__(self, input_dim, mid_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.conv = nn.Conv3d(
            in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, padding=1, groups=mid_dim
        )
        self.linear2 = nn.Linear(mid_dim, input_dim)

    def forward(self, features):  # 1,32,32,32,768
        out = self.linear1(features)
        out = F.relu(out)
        out = out.permute(0, 4, 1, 2, 3)
        out = self.conv(out)
        out = out.permute(0, 2, 3, 4, 1)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = features + out
        return out  # 1,32,32,32,768


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class ImageEncoderViT_3d(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        patch_depth: int = 32,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        cubic_window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        num_slice=1,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_slice = num_slice
        if self.num_slice > 1:
            self.slice_embed = nn.Conv3d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=(1, 1, self.num_slice),
                stride=(1, 1, self.num_slice),
                groups=embed_dim,
            )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
            self.depth_embed = nn.Parameter(torch.zeros(1, patch_depth, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block_3d(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=cubic_window_size,
                res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                shift=cubic_window_size // 2 if i % 2 == 0 else 0,
            )
            self.blocks.append(block)

        self.neck_3d = nn.ModuleList()
        for i in range(4):
            self.neck_3d.append(
                nn.Sequential(
                    nn.Conv3d(768, out_chans, 1, bias=False),
                    nn.InstanceNorm3d(out_chans),
                    nn.ReLU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.patch_embed(x)
        if self.num_slice > 1:
            x = self.slice_embed(x.permute(3, 1, 2, 0).unsqueeze(0))
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(1, 2, 0, 3).unsqueeze(0)

        if self.pos_embed is not None:
            pos_embed = (
                F.avg_pool2d(self.pos_embed.permute(0, 3, 1, 2), kernel_size=2)
                .permute(0, 2, 3, 1)
                .unsqueeze(3)
            )
            pos_embed = pos_embed + (self.depth_embed.unsqueeze(1).unsqueeze(1))
            x = x + pos_embed

        # print("x in img embedding", x.size())
        idx = 0
        feature_list = []
        for blk in self.blocks[:6]:
            x = blk(x)
            idx += 1
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx // 3 - 1](x.permute(0, 4, 1, 2, 3)))
        for blk in self.blocks[6:12]:
            x = blk(x)
            idx += 1
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx // 3 - 1](x.permute(0, 4, 1, 2, 3)))

        x = self.neck_3d[-1](x.permute(0, 4, 1, 2, 3))
        return x, feature_list


class ImageEncoderViT_3d_v2(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        patch_depth: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        cubic_window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        num_slice=1,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        # 初始化PatchEmbed來將影像切割成多個小塊並進行嵌入
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),  # 16, 16
            stride=(patch_size, patch_size),  # 16, 16
            in_chans=in_chans,  # 3
            embed_dim=embed_dim,  # 768
        )

        self.num_slice = num_slice  # 16
        self.embed_dim = embed_dim  # 768
        # 如果num_slice大於1,則使用nn.Conv3d來進行切片嵌入
        if self.num_slice > 1:
            self.slice_embed = nn.Conv3d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=(1, 1, self.num_slice),
                stride=(1, 1, self.num_slice),
                groups=embed_dim,
            )

        # 初始化絕對位置嵌入
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
            self.depth_embed = nn.Parameter(torch.ones(1, patch_depth, embed_dim))

        # 創建多個Block_3d來構成Transformer的主體
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block_3d(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=cubic_window_size,
                res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                shift=cubic_window_size // 2 if i % 2 == 0 else 0,
            )
            self.blocks.append(block)

        # 創建一個nn.ModuleList來實現3D的頸部結構
        self.neck_3d = nn.ModuleList()
        for i in range(4):
            self.neck_3d.append(
                nn.Sequential(
                    nn.Conv3d(768, out_chans, 1, bias=False),
                    LayerNorm3d(out_chans),
                    nn.Conv3d(
                        out_chans,
                        out_chans,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    LayerNorm3d(out_chans),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # logging.info("""
        #              ### image encoder ###
        #              """)
        with torch.no_grad():
            # logging.info(f"輸入的影像維度x: {list(x.size())}")
            x = self.patch_embed(x)  # 將影像切割成多個小塊並進行嵌入,[512, 3, 512, 512]=>[512, 32, 32, 768]
            # logging.info(f"x經過SAM的patch_embed(包含conv2D[3->768]以及permute): {list(x.size())}")

        if self.num_slice > 1:  # 16
            # [1, 768, 32, 32, 512] => slice_embed => [1,768,32,32,32]
            x = x.permute(3, 1, 2, 0).unsqueeze(0)
            # logging.info(f"x經過permute & unsqueeze: {list(x.size())}")
            x = self.slice_embed(x)  # 使用nn.Conv3d來進行切片嵌入
            # logging.info(f"x經過slice_embed(conv3D): {list(x.size())}")
            # logging.info(f"""
            #              in_channels={self.embed_dim},
            #              out_channels={self.embed_dim},
            #              kernel_size={(1,1,self.num_slice)},
            #              stride={(1,1,self.num_slice)},
            #              groups={self.embed_dim}
            #              """)

            x = x.permute(0, 2, 3, 4, 1)  # [1,32,32,32,768]
            # logging.info(f"x經過permute: {list(x.size())}")

        else:
            x = x.permute(1, 2, 0, 3).unsqueeze(0)

        if self.pos_embed is not None:  # 如果使用絕對位置嵌入
            # logging.info("""
            #              ###使用絕對位置嵌入###
            #              """)

            # logging.info(
            #     f"self pos_embed(一組可訓練的零參數但實際上不是零@@): {list(self.pos_embed.size())}"
            # )  # 1,64,64,768

            pos_embed = F.avg_pool2d(self.pos_embed.permute(0, 3, 1, 2), kernel_size=4)
            # logging.info(f"pos_embed經過permute & pool2d: {list(pos_embed.size())}")  # 1,768,32,32

            pos_embed = pos_embed.permute(0, 2, 3, 1).unsqueeze(3)
            # logging.info(
            #     f"pos_embed經過permute & unsqueeze: {list(pos_embed.size())}"
            # )  # 1,32,32,1,768

            # logging.info(
            #     f"self depth_embed(一組可訓練的壹參數): {list(self.depth_embed.unsqueeze(1).unsqueeze(1).size())}"
            # )
            pos_embed = pos_embed + (self.depth_embed.unsqueeze(1).unsqueeze(1))  # 計算位置嵌入
            # logging.info(
            #     f"pos_embed與self depth_embed相加: {list(pos_embed.size())}"
            # )  # [1, 32, 32, 32, 768]

            x = x + pos_embed
            # logging.info(f"影像特徵x與pos_embed相加: {list(x.size())}")  # [1, 32, 32, 32, 768]

        idx = 0
        feature_list = []
        # print("x in img embedding", x.size())  # [1, 32, 32, 32, 768]

        # logging.info("""
        #              ### 開始進入block ###
        #              """)
        for blk in self.blocks[:6]:
            # print("################")
            # print_gpu_utilization()
            x = blk(x)
            # logging.info(f"x經過block: {list(x.size())}")
            idx += 1
            if idx % 3 == 0 and idx != 12:
                # logging.info(f"添加經過neck_conv3D & permute處理的結果到feature list中")
                temp = self.neck_3d[idx // 3 - 1](x.permute(0, 4, 1, 2, 3))
                # logging.info(f"添加的特徵: {list(temp.size())}")
                feature_list.append(temp)
        for blk in self.blocks[6:12]:
            x = blk(x)
            # logging.info(f"x經過block: {list(x.size())}")
            idx += 1
            if idx % 3 == 0 and idx != 12:
                # logging.info(f"添加經過neck_conv3D & permute處理的結果到feature list中")
                temp = self.neck_3d[idx // 3 - 1](x.permute(0, 4, 1, 2, 3))
                # logging.info(f"添加的特徵: {list(temp.size())}")
                feature_list.append(temp)

        x = self.neck_3d[-1](x.permute(0, 4, 1, 2, 3))
        # logging.info(f"x經過neck_conv3D & permute處理: {list(x.size())}")
        # logging.info("""
        #              ###完成image encoder, 回傳影像特徵x、中間層特徵列表feature_list###
        #              """)
        return x, feature_list


class Block_3d(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        res_size=None,
        shift=None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_3d(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=(window_size, window_size, window_size),
            res_size=(res_size, res_size, res_size),
        )
        self.shift_size = shift
        if self.shift_size > 0:
            H, W, D = 16, 16, 16  # 輸入張量的高度、寬度和深度
            img_mask = torch.zeros(
                (1, H, W, D, 1)
            )  # 創建一個形狀為[1, H, W, D, 1]的零張量

            # 定義滑動窗口的移位範圍
            h_slices = (
                slice(0, -window_size),
                slice(-window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -window_size),
                slice(-window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            d_slices = (
                slice(0, -window_size),
                slice(-window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            cnt = 0
            # 對每個窗口進行標籤
            for h in h_slices:
                for w in w_slices:
                    for d in d_slices:
                        img_mask[:, h, w, d, :] = cnt
                        cnt += 1
                        # :, 0:24, 0:24, 0:24, : = 0
                        # :, 0:24, 0:24, 24:28, : = 1
                        # 以此類推,重複之處會被覆蓋

            # 將圖像遮罩劃分為窗口
            mask_windows = window_partition(img_mask, window_size)[0]
            mask_windows = mask_windows.view(-1, window_size * window_size * window_size)

            # 創建注意力遮罩
            # 64,512,512 = [64,1,512]-[64,512,1]
            """
            如果兩個像素點的遮罩值相同(即它們都屬於同一個物體或者都屬於背景), 
            那麼對應的元素值就會是0, 否則, 它會是兩個遮罩值之間的差值。
            這種設計使得模型在計算注意力權重時, 能夠更加關注同一物體內部的像素點, 而忽略不同物體之間的像素點。希望這對您有所幫助！
            """
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None

        # 將注意力遮罩註冊為緩衝區
        self.register_buffer("attn_mask", attn_mask)

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size
        self.adapter = Adapter(input_dim=dim, mid_dim=dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = checkpoint(self.adapter,x)
        x = self.adapter(x)  # linear > 3D > linear
        # logging.info(f"x經過adapter: {list(x.size())}")
        shortcut = x
        # logging.info(f"保存shortcut=x")

        x = self.norm1(x)
        # logging.info(f"x經過norm: {list(x.size())}")

        # Window partition
        if self.window_size > 0:
            H, W, D = x.shape[1], x.shape[2], x.shape[3]
            if self.shift_size > 0:
                # logging.info(f"x在dim(1,2,3)經過roll平移{-self.shift_size}: {list(x.size())}")
                x = torch.roll(
                    x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3)
                )
            x, pad_hw = window_partition(x, self.window_size)  # window_size=8
            # logging.info(f"x經過window_partition: {list(x.size())}")

        x = self.attn(x, mask=self.attn_mask)
        # x = checkpoint(self.attn,x,self.attn_mask)

        # logging.info(f"x經過attention: {list(x.size())}")
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W, D))
            # logging.info(f"x經過window_unpartition: {list(x.size())}")
        if self.shift_size > 0:
            x = torch.roll(
                x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3)
            )
            # logging.info(f"x在dim(1,2,3)經過roll平移{self.shift_size}: {list(x.size())}")

        x = shortcut + x  # skip connection

        # x = x + checkpoint(self.mlp,self.norm2(x))
        x = x + self.mlp(self.norm2(x))
        # logging.info(f"x經過norm & mlp & skipconnection: {list(x.size())}")
        # logging.info("""
        #              ###完成block, 回傳x###
        #              """)
        return x


class Attention_3d(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        res_size=None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每個投平分維度 768//12=
        self.scale = head_dim**-0.5  # 64**-0.5=0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * res_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * res_size[1] - 1, head_dim))
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * res_size[2] - 1, head_dim))
            self.lr = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # logging.info("""
        #              ###attention###
        #              """)
        B, H, W, D, _ = x.shape # 64, 8, 8, 8, 768
        # qkv with shape (3, B, nHead, H * W* D, C)
        qkv = self.qkv(x).reshape(B, H * W * D, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        # q, k, v with shape (B * nHead, H * W * D, C)
        q, k, v = qkv[0], qkv[1], qkv[2] # 64,12,8*8*8,64
        
        # q, k, v = qkv.reshape(3, B * self.num_heads, H * W * D, -1).unbind(0)
        q_sub = q.reshape(B * self.num_heads, H * W * D, -1) # [768, 512, 64]

        # 跟原始SAM不同, 只有k做B * self.num_heads
        # q_sub: [768 ,512, 64]
        # k: [64, 12 ,512, 64]
        # v: [64, 12 ,512, 64]
        # logging.info(f"q: {list(q.size())}")
        # logging.info(f"q_sub: {list(q_sub.size())}")
        # logging.info(f"k: {list(k.size())}")
        # logging.info(f"v: {list(v.size())}")

        attn = (q * self.scale) @ k.transpose(-2, -1)
        # logging.info(f"attn = q@k: {list(attn.size())}")

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn,
                q_sub,
                self.rel_pos_h,
                self.rel_pos_w,
                self.rel_pos_d,
                (H, W, D),
                (H, W, D),
                self.lr,
            )
            attn = attn.reshape(B, self.num_heads, H * W * D, -1)
            # logging.info(f"attn 經過相對位置編碼: {list(attn.size())}")

        if mask is None:
            attn = attn.softmax(dim=-1)
        else:
            nW = mask.shape[0] # 64
            # print("mask.unsqueeze(1).unsqueeze(0)", mask.unsqueeze(1).unsqueeze(0).size())
            # print(
            #     "B, B // nW , nW, self.num_heads, H*W*D, H*W*D",
            #     B,
            #     B // nW,
            #     nW,
            #     self.num_heads,
            #     H * W * D,
            #     H * W * D,
            # )
            # print("attn", attn.view(B // nW, nW, self.num_heads, H * W * D, H * W * D).size())
            attn = attn.view(B // nW, nW, self.num_heads, H * W * D, H * W * D) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, H * W * D, H * W * D)
            attn = attn.softmax(dim=-1)
            # logging.info(f"attn 經過attn_mask, view, softmax: {list(attn.size())}")

        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, D, -1)
            .permute(0, 2, 3, 4, 1, 5)
            .reshape(B, H, W, D, -1)
        )
        
        x = self.proj(x)
        # logging.info(f"x = attn@v再reshape: {list(x.size())}")
        # logging.info("""
        #              ###end of attention###
        #              """)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    將輸入劃分為非重疊的窗口,如果需要,會在劃分之前進行填充。
    Args:
        x (tensor): 輸入張量,形狀為[B, H, W, C]。
        window_size (int): 窗口大小。
    Returns:
        windows: 劃分後的窗口,形狀為[B * num_windows, window_size, window_size, C]。
        (Hp, Wp): 劃分前的填充高度和寬度
    """
    B, H, W, D, C = x.shape

    """
    具體來說, H % window_size 會計算出圖像的高度 H 與窗口大小 window_size 的餘數,
    這個餘數實際上就是最後一個窗口超出的部分。然後,window_size - H % window_size 會計算出需要填充的部分,
    以使得圖像的高度可以被 window_size 整除。

    然而,如果圖像的高度已經可以被 window_size 整除,
    那麼 H % window_size 就會等於0,這時候 window_size - H % window_size 就會等於 window_size,
    也就是說,會多出一個窗口的填充。
    為了避免這種情況,我們再對 window_size - H % window_size 取一次 window_size 的餘數,
    這樣就可以確保當 H 可以被 window_size 整除時,填充的大小為0
    """
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    pad_d = (window_size - D % window_size) % window_size
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        # 最後一個維度前後填充0, 倒數第2個維度後面填充pad_d, 倒數第3個維度後面填充pad_w, 倒數第4個維度後面填充pad_h
        # x 會變成 [1,32+pad_h,32+pad_w,32+pad_d,1]
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    # [1, 32, 32, 32, 1] => [1, 4, 8, 4, 8, 4, 8, 1]
    x = x.view(
        B,
        Hp // window_size,
        window_size,
        Wp // window_size,
        window_size,
        Dp // window_size,
        window_size,
        C,
    )

    # [1, 4, 8, 4, 8, 4, 8, 1] => [1, 4, 4, 4, 8, 8, 8, 1] => [64, 8, 8, 8, 1]
    # 其中 64 是窗口的數量(也就是 4*4*4) , 8 是窗口的大小, 1 是通道數量。
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, window_size, window_size, window_size, C)
    )
    # 類似patch
    return windows, (Hp, Wp, Dp)  # 填充過的大小


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int, int], hw: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp, Dp = pad_hw
    H, W, D = hw
    B = windows.shape[0] // (
        Hp * Wp * Dp // window_size // window_size // window_size
    )  # 變回原本的batch
    x = windows.view(
        B,
        Hp // window_size,
        Wp // window_size,
        Dp // window_size,
        window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hp, Wp, Dp, -1)

    if Hp > H or Wp > W or Dp > D:  # 去掉之前的padding,
        x = x[:, :H, :W, :D, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).
    Returns:
        Extracted positional embeddings according to relative positions.
    """
    # 計算最大的相對距離
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # 如果相對位置嵌入的大小不等於最大的相對距離, 則使用插值方法來調整相對位置嵌入的大小
    if rel_pos.shape[0] != max_rel_dist:
        # 使用線性插值調整相對位置嵌入的大小
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        # 如果相對位置嵌入的大小等於最大的相對距離, 則不需要調整大小
        rel_pos_resized = rel_pos

    # 根據query和key的坐標來計算相對坐標
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    
    # 從調整大小後的相對位置嵌入中選取相應的嵌入
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    rel_pos_d: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
    lr,
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w, q_d = q_size
    k_h, k_w, k_d = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    Rd = get_rel_pos(q_d, k_d, rel_pos_d)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, q_d, dim)
    rel_h = torch.einsum("bhwdc,hkc->bhwdk", r_q, Rh)
    rel_w = torch.einsum("bhwdc,wkc->bhwdk", r_q, Rw)
    rel_d = torch.einsum("bhwdc,dkc->bhwdk", r_q, Rd)

    attn = (
        attn.view(B, q_h, q_w, q_d, k_h, k_w, k_d)
        + lr * rel_h[:, :, :, :, :, None, None]
        + lr * rel_w[:, :, :, :, None, :, None]
        + lr * rel_d[:, :, :, :, None, None, :]
    ).view(B, q_h * q_w * q_d, k_h * k_w * k_d)

    return attn
