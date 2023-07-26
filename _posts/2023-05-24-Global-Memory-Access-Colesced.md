## 线程索引的计算

一个三维数组中的下标是：$(x, y, z)$，这三个维度的大小分别是：$(D_x, D_y, D_z)$。假设这个三维数组以行优先方式存储，那么这个三维坐标的一维表示是：$\text{id} = z *D_x*D_y+y*D_x+x$。

在CUDA编程模型中，一个CTA可以用$\text{threadIdx.x}$, $\text{threadIdx.y}$, $\text{threadIdx.z}$ 三个逻辑坐标去索引。可以把一个CTA内的线程编号想象成一个的三维的整型数组，这个数组的形状是：$[\text{blockDim.x}, \text{blockDim.y}, \text{blockDim.z}]$， 三个维度的跨度（stride）是 $[1, \text{blockDim.x}, \text{blockDim.y}*\text{blockDim.z}]$，于是给定一个线程在线程块内的坐标：$(x, y, z)$，对应的线程块内的一维线程坐标是：

$$\text{threadId} = x+y*\text{blockDim.x}+z*\text{blockDim.y}*\text{blockDim.z}$$

<p align="center"><img src="/images/threads_indices.png" width="30%"/><br>Fig.1 线程组织</p>

线程块也可以以三维坐标进行索引。将三维线程块坐标转换为一维坐标原理相同，最终一个线程的全局一维线程id为：
$$\text{tid}=\text{BlockId} * \text{BlockSize} + \text{ThreadId}$$
其中，$\text{BlockId}$：当前block在grid中的1维坐标。$\text{BlockSize}$：一个block中含有多少个线程。$\text{ThreadId}$：当前线程在线程块中的一维坐标。

|Example|BlockSize|BlockId|ThreadId|ID|
|:--|:--|:--|:--|:--|
|1D Grid, 1D block|$\text{blockDim}.x$|$\text{blockIdx}.x$|$\text{threadIdx}.x$|$\text{blockIdx}.x * \text{blockDim}.x+\text{threadIdx}.x$|
|2D Grid, 1D block|$\text{blockDim}.x * \text{blockDim}.y$|$\text{blockIdx}.y * \text{gridDim.x}+ \text{blockIdx}.x$|$\text{threadIdx}.x$|$\text{blockId} * \text{blockSize} + \text{ThreadIdx}$
|3D grid, 3D block|$blockDim.x * blockDim.y * blockDim.z$|$blockIdx.z * \text{gridDim}.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x$|$threadIdx.z * blockDim.y * \text{blockDim}.x +threadIdx.y * blockDim.x+ threadIdx.x$|$\text{BlockSize}*\text{BlockId}+\text{ThreadId}$|

线程编号连续的32个线程为一个线程束。一个线程束内的线程访问Global Memory时，如果内存地址是连续的，访问将会被合并成内存事务。