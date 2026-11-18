"""Cài đặt các phép toán CNN "từ con số 0" bằng NumPy — phục vụ mục đích giảng giải.

Mỗi hàm bám sát công thức trong Chương 9 của [1] Goodfellow et al. (2016):

* conv2d     — cross-correlation, Eq. (9.6), §9.1, tr. 333
               S(i, j) = sum_m sum_n I(i+m, j+n) K(m, n)
* conv2d_multi — tích chập đa kênh, Eq. (9.7), §9.5, tr. 348
               Z[i, j, k] = sum_{l,m,n} V[l, j+m, k+n] K[i, l, m, n]
* relu       — tầng "detector", §9.3 (Hình 9.7), §6.3.1, tr. 191–193
* max_pool2d — tầng gộp (pooling), §9.3, tr. 339–345
* output_size — kích thước đầu ra với padding P và stride S, §9.5

Không dùng cho huấn luyện (chậm); dùng để minh hoạ trên slide và kiểm chứng
với torch.nn.functional.conv2d.
"""
from __future__ import annotations

import numpy as np


def output_size(w: int, k: int, p: int = 0, s: int = 1) -> int:
    """O = floor((W - K + 2P) / S) + 1.

    Với P = 0, S = 1 ta có "valid convolution": O = W - K + 1 ([1, §9.5, tr. 351]).
    """
    return (w - k + 2 * p) // s + 1


def conv2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Cross-correlation 2D một kênh (thứ mà các thư viện DL gọi là 'convolution')."""
    if padding:
        image = np.pad(image, padding, mode="constant")
    h, w = image.shape
    kh, kw = kernel.shape
    oh, ow = output_size(h, kh, 0, stride), output_size(w, kw, 0, stride)
    out = np.zeros((oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            patch = image[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(patch * kernel)   # tương tác thưa: chỉ kh*kw điểm ảnh
    return out


def conv2d_multi(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None,
                 stride: int = 1, padding: int = 0) -> np.ndarray:
    """Tích chập đa kênh.

    x:      (C_in, H, W)
    weight: (C_out, C_in, K, K)   — cùng một bộ trọng số dùng cho mọi vị trí
                                     (chia sẻ tham số, [1, §9.2]).
    return: (C_out, H_out, W_out)
    """
    c_out, c_in, kh, kw = weight.shape
    assert x.shape[0] == c_in, "Số kênh vào không khớp"
    if padding:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    _, h, w = x.shape
    oh, ow = output_size(h, kh, 0, stride), output_size(w, kw, 0, stride)
    out = np.zeros((c_out, oh, ow), dtype=np.float32)
    for o in range(c_out):
        for i in range(oh):
            for j in range(ow):
                patch = x[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
                out[o, i, j] = np.sum(patch * weight[o])
        if bias is not None:
            out[o] += bias[o]
    return out


def relu(x: np.ndarray) -> np.ndarray:
    """g(z) = max(0, z) — Rectified Linear Unit."""
    return np.maximum(0, x)


def max_pool2d(x: np.ndarray, size: int = 2, stride: int = 2) -> np.ndarray:
    """Max pooling: giữ giá trị lớn nhất trong mỗi vùng size×size.

    Giúp biểu diễn bất biến xấp xỉ với các phép tịnh tiến nhỏ ([1, §9.3, tr. 342]).
    """
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
    c, h, w = x.shape
    oh, ow = output_size(h, size, 0, stride), output_size(w, size, 0, stride)
    out = np.zeros((c, oh, ow), dtype=x.dtype)
    for i in range(oh):
        for j in range(ow):
            out[:, i, j] = x[:, i * stride:i * stride + size, j * stride:j * stride + size].max(axis=(1, 2))
    return out[0] if squeeze else out


def softmax(z: np.ndarray) -> np.ndarray:
    """softmax(z)_i = exp(z_i) / sum_j exp(z_j)  ([1, §6.2.2.3, tr. 184])."""
    z = z - z.max()                       # ổn định số học
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Ví dụ: phát hiện cạnh dọc — tái hiện Hình 9.6 của [1] (tr. 340).
# Kernel 2 phần tử [-1, 1]: mỗi điểm ảnh trừ đi điểm ảnh bên trái nó.
# Ảnh 320×280 → 319×280, chỉ tốn 319·280·3 = 267 960 phép toán dấu phẩy động.
# ---------------------------------------------------------------------------
EDGE_KERNEL_VERTICAL = np.array([[-1.0, 1.0]], dtype=np.float32)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T.copy()


def demo() -> None:
    rng = np.random.default_rng(0)
    img = np.zeros((8, 8), dtype=np.float32)
    img[:, 4:] = 1.0                       # ảnh có một cạnh dọc ở giữa
    edges = conv2d(img, SOBEL_X, padding=1)
    print("Ảnh vào:\n", img)
    print("Bản đồ đặc trưng (Sobel-x):\n", edges)
    print("Sau ReLU + MaxPool 2x2:\n", max_pool2d(relu(edges)))

    x = rng.standard_normal((3, 16, 16)).astype(np.float32)
    w = rng.standard_normal((8, 3, 3, 3)).astype(np.float32)
    y = conv2d_multi(x, w, padding=1)
    print("Đa kênh:", x.shape, "->", y.shape, "| số tham số:", w.size)


if __name__ == "__main__":
    demo()
