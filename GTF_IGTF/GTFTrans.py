import torch
from PIL import Image
from einops import rearrange
from torch import nn
from torchvision import transforms

from .filters import filters1, filters2
import matplotlib.pyplot as plt
import torch.nn.functional as F


def GTF_init(x, filters):

    # if not isinstance(filters, list):
    #     filters = [filters]

    # 维度转换
    if x.dim() == 2:  # 输入的维度是 [rows, cols]
        x = x.unsqueeze(0).unsqueeze(0).float()  # 变为 [1, 1, rows, cols]
    if x.dim() == 3:  # 输入的维度是 [batch, rows, cols]
        x = x.unsqueeze(1).float()  # 变为 [batch, 1, rows, cols]

    result = []
    for f in filters:
        # 将滤波器 f 转换为 tensor，并扩展为形状 [1, 1, k, k]
        kernel = torch.tensor(f, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        k = kernel.shape[-1]
        pad = k // 2
        padded_x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        conv_result = F.conv2d(padded_x, kernel)
        result.append(conv_result)
    # [batch, 9, W, H]
    return torch.cat(result, dim=1)


def IGTF_init(coe, filters):

    # if not isinstance(filters, list):
    #     filters = [filters]

    # test [1, 9, 512, 512]
    batch, Fnum, H, W = coe.shape
    # 创建一个与卷积结果相同形状的输出（注意包含通道维度）
    output = torch.zeros((batch, 1, H, W), dtype=coe.dtype, device=coe.device)
    for i in range(Fnum):
        # 将滤波器 f 转换为 tensor，并扩展为形状 [1, 1, k, k]
        kernel = torch.tensor(filters[i], dtype=coe.dtype, device=coe.device).unsqueeze(0).unsqueeze(0)
        k = kernel.shape[-1]
        pad = k // 2
        conv_result = F.conv2d(
            F.pad(coe[:, i:i + 1, :, :], (pad, pad, pad, pad), mode='circular'),
            kernel
        )
        # conv_result 的形状为 [batch, 1, H, W], 将通道维度去掉后累加到 output 上
        output += conv_result

    return output


def show_results(results, IGTF_res):
    # 移除batch
    results_np = results.squeeze(0).cpu().numpy()  # [F, H, W]
    IGTF_res_np = IGTF_res.squeeze(0).squeeze(0).cpu().numpy()
    n = results_np.shape[0]  # 滤波器的数量 F

    # 设置显示的行数和列数
    rows = 6
    cols = (n // rows) + (1 if n % rows != 0 else 0)

    # 为了显示 IGTF 结果，多加一列：cols + 1
    fig, axes = plt.subplots(rows, cols + 1, figsize=(12, 12))
    axes = axes.flatten()

    # 循环显示每个滤波器结果
    for i in range(n):
        axes[i].imshow(results_np[i, :, :], cmap='gray')
        axes[i].set_title(f"Filter {i + 1}")
        axes[i].axis('off')

    # 显示 IGTF_res（假设 IGTF_res 为 numpy 数组或已转换为可显示格式）
    axes[n].imshow(IGTF_res_np, cmap='gray')
    axes[n].set_title("IGTF Result")
    axes[n].axis('off')

    # 如果有多余的子图，则隐藏它们
    for i in range(n + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


class GTF(nn.Module):
    def __init__(self):
        super(GTF, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return GTF_init(x, filters1)


class IGTF(nn.Module):
    def __init__(self):
        super(IGTF, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return IGTF_init(x, filters2)


if __name__ == '__main__':
    image_path = 'img/piglet.png'
    img = Image.open(image_path)

    # test GTF_transform
    img = img.convert('L')
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img)
    img_tensor = rearrange(img_tensor.float(), 'b h w -> b 1 h w')
    gtf = GTF()
    output = gtf(img_tensor)
    # output = output[:, 0:1, :, :]
    print("Output shape:", output.shape)
    igtf = IGTF()
    IGTF_res = igtf(output)
    print("IGTF_res shape:", IGTF_res.shape)
    show_results(output, IGTF_res)

    # x = torch.randn(1, 1, 512, 512)  # 测试图像
    # coe = GTF()(x)  # 分解
    # x_recon = IGTF()(coe)  # 重构
    # print("Reconstruction MSE:", F.mse_loss(output, IGTF_res).item())  # 理想值应接近0
