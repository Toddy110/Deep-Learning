# 目录

- [1. 图像基础与本质](#1-图像基础与本质)
    - [1.1 图像的输入结构](#11-图像的输入结构)
    - [1.2 数字图像的本质](#12-数字图像的本质)
    - [1.3 栅格图像与矢量图像](#13-栅格图像与矢量图像)
- [2. 卷积操作与特征提取](#2-卷积操作与特征提取)
    - [2.1 卷积操作的基本原理](#21-卷积操作的基本原理)
    - [2.2 卷积层的结构与流程](#22-卷积层的结构与流程)
- [3. 卷积操作的数学与代码示例](#3-卷积操作的数学与代码示例)
    - [3.1 单通道卷积计算过程](#31-单通道卷积计算过程)
    - [3.2 多通道卷积的结构与参数](#32-多通道卷积的结构与参数)
    - [3.3 代码示例：多通道卷积](#33-代码示例多通道卷积)
- [4. 卷积操作的参数详解](#4-卷积操作的参数详解)
    - [4.1 Padding（填充）](#41-padding填充)
    - [4.2 Stride（步幅）](#42-stride步幅)
- [5. 下采样（Pooling）操作](#5-下采样pooling操作)
    - [5.1 MaxPooling（最大池化）](#51-maxpooling最大池化)
- [6. CNN网络结构流程举例](#6-cnn网络结构流程举例)

# CNN (Convolutional Neural Network, 卷积神经网络)（基础）

---

## 1. 图像基础与本质

### 1.1 图像的输入结构
以MNIST数据集为例，灰度图像的输入尺寸为 $1 \times 28 \times 28$，即：
- 通道数（Channel）：1，表示灰度图像只有一个颜色通道；
- 宽度（Width）：28，图像的宽度为28像素；
- 高度（Height）：28，图像的高度为28像素。

在实际应用中，彩色图像通常包含红（R）、绿（G）、蓝（B）三个通道（channel）。

### 1.2 数字图像的本质
数字图像（digital image）本质上是由像素（Pixel）组成的二维矩阵，每个像素记录了对应位置的光强或颜色信息。
- 在数字成像系统（digital imaging system）中，常用光敏电阻（Photoresistor）或光电二极管（Photodiode）作为感光元件。光敏电阻的电阻值会随着光照强度的变化而变化。
- 通过在电路中测量光敏电阻两端的电压和电流，可以计算出其电阻值，进而推算出该位置的光强。
- 成像时，通常会使用透镜系统将外部光线聚焦到感光元件阵列上。每个光敏电阻对应一个光锥，接收器尺寸越小，光锥角度越小，能够采集的空间分辨率越高。
- 多个光敏电阻按照规则排布，形成感光阵列，每个光敏电阻即为一个像素。整个阵列即可采集一幅完整的图像。
- 彩色图像通常在每个像素位置上设置多个不同颜色滤光片（如红、绿、蓝），分别采集不同波段的光强，实现彩色成像。
- 在实际的彩色数字成像设备中，最常见的滤色片排列方式是拜耳阵列（Bayer Pattern），其中最常见的是RGGB排列。RGGB表示每2×2的像素块中有2个绿色、1个红色和1个蓝色滤光片，排列如下：
  - 红（R） 绿（G）
  - 绿（G） 蓝（B） \
  这种设计是因为人眼对绿色更敏感，因此绿色像素数量更多。通过拜耳阵列采集到的原始数据，经过插值算法（去马赛克，Demosaicing）后，可以还原出全分辨率的彩色图像。

因此，数字图像的本质是对空间中光强分布的离散采样和数字化表达。分辨率越高，能够还原的细节越丰富。

### 1.3 栅格图像与矢量图像
- **栅格图像（Raster Image）**：由像素（Pixel）组成的二维矩阵，每个像素有固定的颜色或灰度值。常见格式有JPEG、PNG、BMP等。优点是能够表现丰富的细节和复杂的色彩变化，适合照片、扫描图像等。缺点是放大后会出现锯齿（失真），文件体积较大。
- **矢量图像（Vector Image）**：由点、线、曲线和多边形等几何图形通过数学公式描述。常见格式有SVG、EPS、PDF等。优点是无论放大多少倍都不会失真，文件体积通常较小，适合图标、标志、插画等。缺点是不适合表现复杂的色彩渐变和细节丰富的照片。

**应用场景**：
- 栅格图像常用于数码摄影、医学影像、遥感等需要表现真实世界细节的场合。
- 矢量图像常用于平面设计、排版、CAD制图等需要高精度缩放和清晰边界的场合。

---

## 2. 卷积操作与特征提取

### 2.1 卷积操作的基本原理
在进行卷积操作时，通常会在图像上选取一个小块（patch），其形状为 $3 \times H' \times W'$，其中3表示输入的通道数（如RGB三通道），$H'$ 和 $W'$ 分别为该块的高度和宽度。这个小块会作为卷积核的感受野，在整张图像上以滑动窗口的方式移动，遍历所有位置。每次滑动时，对当前区域进行卷积计算，得到一个输出值。所有位置的输出值拼接在一起，形成新的特征图（feature map）。卷积操作完成后，输出的通道数、宽度和高度通常会发生变化。每个输出通道（output channel）都综合了输入patch中的全部信息，代表了不同的特征响应。

### 2.2 卷积层的结构与流程
- 图像首先经过卷积层（convolutional layer），如 $5 \times 5$ 卷积核，可以得到4个通道、每个通道 $24 \times 24$ 的特征图（feature map，记作 $C_1$）。卷积层的作用是提取图像的空间特征。卷积操作后，输出张量依然保持三维结构（通道数 × 宽度 × 高度），但通道数、宽度和高度可能发生变化。
- 与全连接层（fully connected layer）不同，全连接层会将图像展平成一维向量，丧失原有的空间结构信息。
- 卷积层输出的特征图通常会经过下采样（subsampling/pooling），如 $2 \times 2$ 池化，得到新的特征图（$S_1$），例如4个通道、每个通道 $12 \times 12$。下采样操作会减小宽度和高度，通道数保持不变，主要目的是减少数据量，降低计算复杂度。
- 可以继续堆叠卷积层和下采样层。例如，再经过一个 $5 \times 5$ 卷积，得到 $C_2$（8个通道，$8 \times 8$），再下采样一次，得到 $S_2$（8个通道，$4 \times 4$）。
- 最后，将三阶张量（如 $8 \times 4 \times 4$）展平成一维向量（通过 view 操作），再通过全连接层映射到十维输出，完成分类任务。常用交叉熵损失（cross-entropy loss）和 softmax 进行概率分布计算。

**总结**：构建神经网络时，需要明确输入和输出张量的维度，并通过不同的层结构将其映射到目标空间，实现特征提取与分类（Feature Extraction + Classification）。

---

## 3. 卷积操作的数学与代码示例

### 3.1 单通道卷积计算过程
以 $5 \times 5$ 的输入矩阵和 $3 \times 3$ 的卷积核为例，演示卷积操作的计算过程：

输入（Input）：

<table style="border-collapse: collapse;">
  <tr>
    <td style="border:2px solid red;"><b>3</b></td>
    <td style="border:2px solid red;"><b>4</b></td>
    <td style="border:2px solid red;"><b>6</b></td>
    <td style="border:2px solid #aaa;">5</td>
    <td style="border:2px solid #aaa;">7</td>
  </tr>
  <tr>
    <td style="border:2px solid red;"><b>2</b></td>
    <td style="border:2px solid red;"><b>4</b></td>
    <td style="border:2px solid red;"><b>6</b></td>
    <td style="border:2px solid #aaa;">8</td>
    <td style="border:2px solid #aaa;">2</td>
  </tr>
  <tr>
    <td style="border:2px solid red;"><b>1</b></td>
    <td style="border:2px solid red;"><b>6</b></td>
    <td style="border:2px solid red;"><b>7</b></td>
    <td style="border:2px solid #aaa;">8</td>
    <td style="border:2px solid #aaa;">4</td>
  </tr>
  <tr>
    <td style="border:2px solid #aaa;">9</td>
    <td style="border:2px solid #aaa;">7</td>
    <td style="border:2px solid #aaa;">4</td>
    <td style="border:2px solid #aaa;">6</td>
    <td style="border:2px solid #aaa;">2</td>
  </tr>
  <tr>
    <td style="border:2px solid #aaa;">3</td>
    <td style="border:2px solid #aaa;">7</td>
    <td style="border:2px solid #aaa;">5</td>
    <td style="border:2px solid #aaa;">4</td>
    <td style="border:2px solid #aaa;">1</td>
  </tr>
</table>

卷积核（Kernel）：

<table style="border-collapse: collapse;">
  <tr>
    <td style="border:2px solid #aaa;">1</td>
    <td style="border:2px solid #aaa;">2</td>
    <td style="border:2px solid #aaa;">3</td>
  </tr>
  <tr>
    <td style="border:2px solid #aaa;">4</td>
    <td style="border:2px solid #aaa;">5</td>
    <td style="border:2px solid #aaa;">6</td>
  </tr>
  <tr>
    <td style="border:2px solid #aaa;">7</td>
    <td style="border:2px solid #aaa;">8</td>
    <td style="border:2px solid #aaa;">9</td>
  </tr>
</table>

输出（Output）：

<table style="border-collapse: collapse;">
  <tr>
    <td style="border:2px solid #aaa;">211</td>
    <td style="border:2px solid #aaa;">295</td>
    <td style="border:2px solid #aaa;">262</td>
  </tr>
  <tr>
    <td style="border:2px solid #aaa;">259</td>
    <td style="border:2px solid #aaa;">282</td>
    <td style="border:2px solid #aaa;">214</td>
  </tr>
  <tr>
    <td style="border:2px solid #aaa;">251</td>
    <td style="border:2px solid #aaa;">253</td>
    <td style="border:2px solid #aaa;">169</td>
  </tr>
</table>

**详细计算过程：**

- **左上角输出（第1行第1列）：**
  $3\times1 + 4\times2 + 6\times3 + 2\times4 + 4\times5 + 6\times6 + 1\times7 + 6\times8 + 7\times9 = 211$

- **第1行第2列：**
  $4\times1 + 6\times2 + 5\times3 + 4\times4 + 6\times5 + 8\times6 + 6\times7 + 7\times8 + 8\times9 = 295$

- **第1行第3列：**
  $6\times1 + 5\times2 + 7\times3 + 6\times4 + 8\times5 + 2\times6 + 7\times7 + 8\times8 + 4\times9 = 262$

- **第2行第1列：**
  $2\times1 + 4\times2 + 6\times3 + 1\times4 + 6\times5 + 7\times6 + 9\times7 + 7\times8 + 4\times9 = 259$

- **第2行第2列：**
  $4\times1 + 6\times2 + 8\times3 + 6\times4 + 7\times5 + 8\times6 + 7\times7 + 4\times8 + 6\times9 = 282$

- **第2行第3列：**
  $6\times1 + 8\times2 + 2\times3 + 7\times4 + 8\times5 + 4\times6 + 4\times7 + 6\times8 + 2\times9 = 214$

- **第3行第1列：**
  $1\times1 + 6\times2 + 7\times3 + 9\times4 + 7\times5 + 4\times6 + 3\times7 + 7\times8 + 5\times9 = 251$

- **第3行第2列：**
  $6\times1 + 7\times2 + 8\times3 + 7\times4 + 4\times5 + 6\times6 + 7\times7 + 5\times8 + 4\times9 = 253$

- **第3行第3列：**
  $7\times1 + 8\times2 + 4\times3 + 4\times4 + 6\times5 + 2\times6 + 5\times7 + 4\times8 + 1\times9 = 169$

> 注：每个输出值对应卷积核在输入矩阵上滑动到不同位置时，patch与kernel对应元素相乘后求和的结果。

### 3.2 多通道卷积的结构与参数
在卷积神经网络中，多通道卷积的本质是：每个输出通道都通过对所有输入通道分别进行卷积、再将结果按元素相加得到。

更为严谨地说，假设输入特征图的通道数为 $C_{in}$，输出特征图的通道数为 $C_{out}$，卷积核的空间尺寸为 $K_h \times K_w$。那么，卷积核的权重张量形状为：
$$
(C_{out},\; C_{in},\; K_h,\; K_w)
$$

具体计算流程如下：

1. **每个输出通道的生成**  
   - 对于每一个输出通道 $o$（$o=1,2,\ldots,C_{out}$），都对应有 $C_{in}$ 个二维卷积核（每个输入通道一个）。
   - 对于每个输入通道 $i$（$i=1,2,\ldots,C_{in}$），用该通道的输入特征图与对应的卷积核进行二维卷积，得到一个中间特征图。
   - 将所有 $C_{in}$ 个中间特征图按元素相加（逐元素求和），得到该输出通道的最终特征图。

2. **卷积核权重的组织**  
   - 整个卷积层的权重是一个四维张量，形状为 $(C_{out},\; C_{in},\; K_h,\; K_w)$。
   - 其中，第 $o$ 个输出通道的卷积核权重为 $W[o, :, :, :]$，包含了对所有输入通道的卷积核。

3. **输出张量的形状**  
   - 假设输入张量形状为 $(N, C_{in}, H_{in}, W_{in})$，其中 $N$ 是批量大小。
   - 输出张量的形状为 $(N, C_{out}, H_{out}, W_{out})$，其中 $H_{out}$ 和 $W_{out}$ 由输入尺寸、卷积核尺寸、步幅、填充等参数共同决定。

4. **举例说明**  
   - 以常见的 RGB 图像为例，$C_{in}=3$。若设置 $C_{out}=16$，则该卷积层共有 $16 \times 3 = 48$ 个二维卷积核，每个输出通道都融合了所有输入通道的信息。

**总结：**
- 每个输出通道都聚合了所有输入通道的卷积结果。
- 卷积核的四维结构确保了输入通道和输出通道之间的全连接。
- 这种设计使得网络能够学习到跨通道的复杂特征组合。

因此，PyTorch 等深度学习框架中，`Conv2d` 层的权重参数形状为 $(\text{out\_channels},\; \text{in\_channels},\; \text{kernel\_size}_{height},\; \text{kernel\_size}_{width})$，严格对应上述数学描述。

### 3.3 代码示例：多通道卷积
由此，我们可以写出卷积操作的示例代码：
``` python
import torch
in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size, in_channels, width, height)

conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size)

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)
```
下面对上述代码中的各个参数和操作进行有条理的说明：

1. **输入输出通道数**
   - `in_channels`：表示输入张量的通道数。例如，对于RGB彩色图像，`in_channels=3`；对于灰度图像，`in_channels=1`。本例中设为5，表示输入有5个通道。
   - `out_channels`：表示卷积操作后输出张量的通道数。这个值由我们自行设定，通常越大，网络能够学习到的特征越丰富。本例中设为10。

2. **图像尺寸**
   - `width` 和 `height`：分别表示输入图像的宽度和高度。本例中均为100。

3. **卷积核大小**
   - `kernel_size`：指定卷积核的空间尺寸。可以是单个整数（表示正方形卷积核），也可以是二元组（如`(3, 5)`，表示非正方形卷积核）。本例中为3，表示 $3 \times 3$ 的卷积核。

4. **输入张量的生成**
   - `input = torch.randn(batch_size, in_channels, width, height)`：生成一个形状为 `(batch_size, in_channels, width, height)` 的四维张量，元素服从标准正态分布。这里 `batch_size=1`，表示一次只输入一张多通道图像。

5. **卷积层的定义**
   - `conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)`：创建一个二维卷积层。需要指定输入通道数、输出通道数和卷积核大小这三个核心参数。
   - **注意**：卷积核的形状不一定要是正方形（即`width=height`），但实际应用中常用正方形卷积核（如 $3 \times 3$、$5 \times 5$ 等），因为其在空间上具有对称性，便于特征提取。

通过上述参数的设定，可以灵活地构建适用于不同输入数据和任务需求的卷积层结构。

输出结果如下：
```
torch.Size([1, 5, 100, 100])
torch.Size([1, 10, 98, 98])
torch.Size([10, 5, 3, 3])
```

---

## 4. 卷积操作的参数详解

### 4.1 Padding（填充）
Padding（填充）用于在输入特征图的边缘补零，以控制输出特征图的空间尺寸。

- 公式：
$$
\text{padding} = \frac{\text{kernel\_size} - 1}{2}
$$
其中 kernel\_size 是卷积核的尺寸（如 3、5、7 等）。
如果卷积核尺寸为奇数，这个公式可以保证输出尺寸与输入尺寸一致。

#### 代码示例：Padding
``` python
import torch

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]

input = torch.Tensor(input).view(1, 1, 5, 5)

conv_layer = torch.nn.Conv2d(1, 1, kernel_size = 3, padding = 1)

kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output)
```

上述代码演示了如何在 PyTorch 中手动设置输入、卷积核，并使用 padding 进行卷积操作。
- 首先，定义了一个 $5 \times 5$ 的输入矩阵，并用 `torch.Tensor(...).view(1, 1, 5, 5)` 变成四维张量，符合卷积层的输入格式（batch size, channel, height, width）。
- 创建了一个 $3 \times 3$ 的卷积核，并手动赋值给卷积层的权重。
- `padding=1` 表示在输入的每一边都补上一圈0，使得输出的空间尺寸与输入一致。
- 最后，将输入送入卷积层，输出结果的空间尺寸仍为 $5 \times 5$，验证了 padding 的作用。

通过这个例子可以直观理解 padding 如何影响卷积输出的尺寸。

输出结果如下：
```
tensor([[[[ 90.9643, 167.9643, 223.9643, 214.9643, 126.9643],
          [113.9643, 210.9643, 294.9643, 261.9643, 148.9643],
          [191.9643, 258.9643, 281.9643, 213.9643, 121.9643],
          [193.9643, 250.9643, 252.9643, 168.9643,  85.9643],
          [ 95.9643, 111.9643, 109.9643,  67.9643,  30.9643]]]],
       grad_fn=<ConvolutionBackward0>)
```

### 4.2 Stride（步幅）
Stride（步幅）用于控制卷积核在输入特征图上每次移动的距离。
- `stride=2` 表示卷积核每次在输入特征图上移动2个像素（而不是默认的1个像素）。
- 这样会导致输出特征图的宽度和高度都变小，相当于对输入做了下采样。
- 通过设置不同的stride，可以灵活控制输出特征图的空间尺寸。
- 本例中，输入为 $5 \times 5$，卷积核为 $3 \times 3$，stride=2，输出的空间尺寸会比stride=1时更小。

#### 代码示例：Stride
``` python
import torch

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]

input = torch.Tensor(input).view(1, 1, 5, 5)

conv_layer = torch.nn.Conv2d(1, 1, kernel_size = 3, stride = 2, bias = False)

kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output)
```

上述代码演示了stride（步幅）参数的作用：
- `stride=2` 表示卷积核每次在输入特征图上移动2个像素。
- 这样会导致输出特征图的宽度和高度都变小，相当于对输入做了下采样。
- 通过设置不同的stride，可以灵活控制输出特征图的空间尺寸。
- 本例中，输入为 $5 \times 5$，卷积核为 $3 \times 3$，stride=2，输出的空间尺寸会比stride=1时更小。

通过这个例子可以直观理解stride对卷积输出尺寸的影响。

输出结果如下：
```
tensor([[[[211., 262.],
          [251., 169.]]]], grad_fn=<ConvolutionBackward0>)
```

---

## 5. 下采样（Pooling）操作

### 5.1 MaxPooling（最大池化）
MaxPooling（最大池化）是一种常用的下采样方法。对于 $2 \times 2 $ 的MaxPooling，默认stride = 2。
- 其原理是在每个 $2 \times 2$ 区域内取最大值，减小特征图尺寸，通道数保持不变。
- Maxpooling只能在一个通道内做，通道之间是无法做Maxpooling的（通道数量不变，图像大小变化）。

#### 代码示例：MaxPooling
``` python
import torch

input = [3, 4, 6, 5,
         2, 4, 6, 8,
         1, 6, 7, 8,
         9, 7, 4, 6]

input = torch.Tensor(input).view(1, 1, 4, 4)

maxpooling_layer = torch.nn.MaxPool2d(kernel_size = 2)

output = maxpooling_layer(input)
print(output)
```
该代码的核心是`torch.nn.MaxPool2d`,并设置`kernel_size = 2`,这样同样也默认了步长stride = 2。

输出结果如下：
```
tensor([[[[4., 8.],
          [9., 8.]]]])
```

---

## 6. CNN网络结构流程举例

- 输入：(batch, 1, 28, 28)
- Conv2d Layer 1: filter $5 \times 5$, $C_{in}$:1, $C_{out}$:10 → (batch, 10, 24, 24)
- Pooling Layer 1: filter $2 \times 2$ → (batch, 10, 12, 12)
- Conv2d Layer 2: filter $5 \times 5$, $C_{in}$:10, $C_{out}$:20 → (batch, 20, 8, 8)
- Pooling Layer 2: filter $2 \times 2$ → (batch, 20, 4, 4)
- 展平成向量，经过全连接层映射为10类输出。

---

最终代码实现：

```python
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='D:/PythonCode/Pytorch_learning/MNIST',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_dataset = datasets.MNIST(
    root='D:/PythonCode/Pytorch_learning/MNIST',
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test() 
```
本笔记系统梳理了卷积神经网络的输入结构、卷积与池化操作的原理、参数设置、数学推导与代码实现，并通过具体的网络结构流程示例，帮助理解CNN的整体信息流与特征提取机制。