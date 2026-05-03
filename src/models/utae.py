"""
U-TAE implementation adapted for this repository from VSainteuf/utae-paps (MIT).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _valid_group_count(num_channels: int, preferred_groups: int) -> int:
    for groups in range(min(num_channels, preferred_groups), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class PositionalEncoder(nn.Module):
    def __init__(self, d: int, T: int = 1000, repeat: int | None = None) -> None:
        super().__init__()
        self.repeat = repeat
        self.register_buffer(
            "denom",
            torch.pow(T, 2 * (torch.arange(d, dtype=torch.float32) // 2) / d),
            persistent=False,
        )

    def forward(self, batch_positions: torch.Tensor) -> torch.Tensor:
        sinusoid_table = batch_positions[:, :, None].float() / self.denom[None, None, :]
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])

        if self.repeat is not None:
            sinusoid_table = torch.cat([sinusoid_table for _ in range(self.repeat)], dim=-1)

        return sinusoid_table


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, attn_dropout: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        attn = self.dropout(self.softmax(attn))
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_k: int, d_in: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k)), requires_grad=True)
        nn.init.normal_(self.Q, mean=0.0, std=math.sqrt(2.0 / d_k))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0.0, std=math.sqrt(2.0 / d_k))

        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_k))

    def forward(
        self,
        v: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, self.d_k)

        k = self.fc1_k(v).view(sz_b, seq_len, self.n_head, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.d_k)

        if pad_mask is not None:
            pad_mask = pad_mask.repeat((self.n_head, 1))

        v = torch.stack(v.split(v.shape[-1] // self.n_head, dim=-1)).view(
            self.n_head * sz_b, seq_len, -1
        )

        output, attn = self.attention(q, k, v, pad_mask=pad_mask)
        attn = attn.view(self.n_head, sz_b, seq_len)
        output = output.view(self.n_head, sz_b, self.d_in // self.n_head)
        return output, attn


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_head: int = 16,
        d_k: int = 4,
        mlp: list[int] | None = None,
        dropout: float = 0.2,
        d_model: int = 256,
        T: int = 1000,
        return_att: bool = False,
        positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.return_att = return_att
        self.n_head = n_head

        if mlp is None:
            mlp = [d_model, in_channels]

        if d_model % n_head != 0:
            raise ValueError("UT-AE requires d_model to be divisible by n_head.")

        self.d_model = d_model
        self.inconv = nn.Conv1d(in_channels, d_model, 1) if d_model != in_channels else None

        if mlp[0] != d_model:
            raise ValueError("The first MLP width must match d_model.")

        self.positional_encoder = (
            PositionalEncoder(d_model // n_head, T=T, repeat=n_head) if positional_encoding else None
        )
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=d_model)
        self.in_norm = nn.GroupNorm(
            num_groups=_valid_group_count(in_channels, n_head),
            num_channels=in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=_valid_group_count(mlp[-1], n_head),
            num_channels=mlp[-1],
        )

        layers: list[nn.Module] = []
        for i in range(len(mlp) - 1):
            layers.extend(
                [
                    nn.Linear(mlp[i], mlp[i + 1]),
                    nn.BatchNorm1d(mlp[i + 1]),
                    nn.ReLU(),
                ]
            )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        sz_b, seq_len, _, h, w = x.shape

        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, self.in_channels)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)
        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        out = self.out_norm(self.dropout(self.mlp(out)))
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)
        if self.return_att:
            return out, attn
        return out


class TemporallySharedBlock(nn.Module):
    def __init__(self, pad_value: float | None = None) -> None:
        super().__init__()
        self.out_shape: torch.Size | None = None
        self.pad_value = pad_value

    def smart_forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 4:
            return self.forward(input)

        b, t, c, h, w = input.shape

        if self.pad_value is not None:
            dummy = torch.zeros(input.shape, device=input.device, dtype=torch.float32)
            self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

        out = input.view(b * t, c, h, w)
        if self.pad_value is not None:
            pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
            if pad_mask.any():
                if self.out_shape is None:
                    raise RuntimeError("UT-AE output shape cache was not initialized.")
                temp = torch.ones(self.out_shape, device=input.device, requires_grad=False) * self.pad_value
                temp[~pad_mask] = self.forward(out[~pad_mask])
                out = temp
            else:
                out = self.forward(out)
        else:
            out = self.forward(out)

        _, c_out, h_out, w_out = out.shape
        return out.view(b, t, c_out, h_out, w_out)


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels: list[int],
        norm: str = "batch",
        k: int = 3,
        s: int = 1,
        p: int = 1,
        n_groups: int = 4,
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d
        elif norm == "group":
            norm_layer = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=_valid_group_count(num_feats, n_groups),
            )
        else:
            norm_layer = None

        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if norm_layer is not None:
                layers.append(norm_layer(nkernels[i + 1]))

            if last_relu or i < len(nkernels) - 2:
                layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels: list[int],
        pad_value: float | None = None,
        norm: str = "batch",
        last_relu: bool = True,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int,
        s: int,
        p: int,
        pad_value: float | None = None,
        norm: str = "batch",
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer([d_in, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer([d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.down(input)
        out = self.conv1(out)
        return out + self.conv2(out)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_skip: int,
        k: int,
        s: int,
        p: int,
        norm: str = "batch",
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d_skip, out_channels=d_skip, kernel_size=1),
            nn.BatchNorm2d(d_skip),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer([d_out + d_skip, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer([d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        return out + self.conv2(out)


class TemporalAggregator(nn.Module):
    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        if mode not in {"att_group", "att_mean", "mean"}:
            raise ValueError(f"Unknown UT-AE aggregation mode: {mode}")
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "mean":
            if pad_mask is None or not pad_mask.any():
                return x.mean(dim=1)
            valid = (~pad_mask).float()[:, :, None, None, None]
            return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        if attn_mask is None:
            raise ValueError("UT-AE attention aggregation requires attention masks.")

        if self.mode == "att_mean":
            attn = attn_mask.mean(dim=0)
            attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
            if pad_mask is not None and pad_mask.any():
                attn = attn * (~pad_mask).float()[:, :, None, None]
            return (x * attn[:, :, None, :, :]).sum(dim=1)

        n_heads, b, t, h, w = attn_mask.shape
        attn = attn_mask.view(n_heads * b, t, h, w)
        if x.shape[-2] > w:
            attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
        else:
            attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
        attn = attn.view(n_heads, b, t, *x.shape[-2:])

        if pad_mask is not None and pad_mask.any():
            attn = attn * (~pad_mask).float()[None, :, :, None, None]

        chunks = torch.stack(x.chunk(n_heads, dim=2))
        out = attn[:, :, :, None, :, :] * chunks
        out = out.sum(dim=2)
        return torch.cat([group for group in out], dim=1)


class UTAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        encoder_widths: list[int],
        decoder_widths: list[int] | None = None,
        out_conv_channels: list[int] | None = None,
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        agg_mode: str = "att_group",
        encoder_norm: str = "group",
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        pad_value: float = 0.0,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        if decoder_widths is None:
            decoder_widths = encoder_widths
        if out_conv_channels is None:
            out_conv_channels = [32]
        if len(encoder_widths) != len(decoder_widths):
            raise ValueError("UT-AE encoder and decoder widths must have the same length.")
        if encoder_widths[-1] != decoder_widths[-1]:
            raise ValueError("UT-AE encoder and decoder bottleneck widths must match.")
        if agg_mode == "att_group" and any(width % n_head != 0 for width in encoder_widths[:-1] + [encoder_widths[-1]]):
            raise ValueError("UT-AE att_group aggregation requires stage widths divisible by n_head.")

        self.pad_value = pad_value
        self.n_stages = len(encoder_widths)

        self.in_conv = ConvBlock(
            nkernels=[input_dim, encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = TemporalAggregator(mode=agg_mode)

        head_channels = [decoder_widths[0], *out_conv_channels]
        if out_conv_channels:
            self.out_conv = ConvBlock(head_channels, last_relu=True, padding_mode=padding_mode)
            classifier_in = out_conv_channels[-1]
        else:
            self.out_conv = nn.Identity()
            classifier_in = decoder_widths[0]
        self.classifier = nn.Conv2d(classifier_in, num_classes, kernel_size=1)

    def forward(
        self,
        input: torch.Tensor,
        batch_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch_positions is None:
            batch_positions = torch.arange(input.shape[1], device=input.device, dtype=torch.float32).unsqueeze(0)
            batch_positions = batch_positions.repeat(input.shape[0], 1)

        pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)

        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        for down_block in self.down_blocks:
            out = down_block.smart_forward(feature_maps[-1])
            feature_maps.append(out)

        out, att = self.temporal_encoder(
            feature_maps[-1],
            batch_positions=batch_positions,
            pad_mask=pad_mask,
        )

        for i, up_block in enumerate(self.up_blocks):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)],
                pad_mask=pad_mask,
                attn_mask=att,
            )
            out = up_block(out, skip)

        out = self.out_conv(out)
        return self.classifier(out)
