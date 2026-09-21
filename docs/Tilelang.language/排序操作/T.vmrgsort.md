# T.vmrgsort

Expert / Ascend910B 的原生 packed 有序列表归并。需要带 `VMrgSortOp` 的 AscendNPU-IR 和对应 `bishengir_mrgsort.aiv.bc`，不支持以旧版 IR 静默降级。

```python
T.vmrgsort(src0, src1, src2, src3, dst,
           element_lengths=(2048, 512, 512, 0),
           valid_bit=7, repeat_times=1, exhausted_suspension=False)
```

- 所有缓冲区在 UB，一维、连续；输入已按 score 降序排序。
- 每个 record 固定8字节：FP32 score + 原始 uint32 index；FP16 则是2字节 score、2字节保留位、4字节 index。长度按 record 计数，不是浮点元素数。
- `element_lengths` 四个编译期整数，各在 `[0,4095]`；`valid_bit` 只能为3、7、15，分别启用前2、3、4路。非活动指针仍须为合法 UB buffer。
- 输出容量必须至少为所有活动长度之和乘 repeat，再乘每条 record 的元素数。输出不得与输入重叠；源地址8字节对齐，目标32字节对齐。
- `repeat_times` 在 `[1,255]`。大于1时，四路等长、连续排在同一个 allocation 中，且不能开启耗尽暂停。每轮输入/输出前进四路合计长度。
- `exhausted_suspension=True`：任何活动队列耗尽即停，只保证实际产生的前缀，其余输出未定义。此接口不返回产生条数；只有已证明所读前缀长度有保证的算法才能使用，例如每路2048条时取前2048条。
- 空活动队列由原生 IR 规范化为不操作、单路复制或压紧非空队列，避免硬件空队列异常。
- 无法静态证明的偏移、容量、实际地址对齐与别名由调用方保证；编译通过不是动态内存安全证明。

## 对应操作

`tl.vmrgsort → hivm.hir.vmrgsort → 同步/内存规划 → ccec 库调用 → vmrgsort4`。

库文件随所选编译器安装在相邻 `lib` 中，TileLang 仅在 IR 使用 packed sort 时链接。普通算子不增加此依赖。A5 和 Developer 模式不在本次接口支持范围内。

## 伴随接口

```python
T.vsort32(scores, indices, packed, repeat_times=16)
# FP32 scores[512] + I32 indices[512] → packed[1024] FP32存储。
# 生成16个独立32-record有序小组，并非整个512排序完成。

T.vextract_pairs(packed, values, indices)
# packed[4096] FP32存储 → values[2048] FP32 + indices[2048] I32。
# 解交织，保留 index 位，不进行 float→int 数值转换。
```

`vsort32` 输入/输出均要求32字节对齐、不可重叠，repeat范围1–255；仅支持 FP32 score 与 I32 index。`vextract_pairs` 组合既有原生 bitcast/deinterleave，无 packed 临时副本。所有接口可接受去掉前导单维后的一维 buffer region。

设备证据分别记录于 LightningIndexer 开发记录；完整 TileLang 接口及性能仍需本轮上板验收，不因原生 MrgSort 单项通过而宣称整个算子通过。
