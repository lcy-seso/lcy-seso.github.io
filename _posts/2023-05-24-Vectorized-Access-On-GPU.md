<!--
 * @Author: Ying
 * @Date: 2023-05-24 08:13:35
 * @Descripttion: 
 * @LastEditors: Ying
 * @LastEditTime: 2023-06-08 10:43:45
-->
# Vectorized Access on GPU

CUDA 里最大支持128 bit（128 bit = 16 bytes，4个浮点数，float4类型） pack 大小。
在浮点数据类型中，最小的类型（half）大小为16 bit，最多能把128 / 16 = 8 个 half 数据 pack 到一起。

# 参考文献

1. [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)