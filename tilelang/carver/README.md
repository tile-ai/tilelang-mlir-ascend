# Carver: A Tile-Structure Based Hint Recommend Framework for Ascend NPU

**Carver** is a lightweight framework for generating and ranking tile configurations (tiling strategies) for Ascend NPU backends. It helps you explore efficient mappings of loops for operations such as matrix multiplication, elementwise transforms, and other reduction-oriented kernels.

Carver combines hardware architecture information, user-defined tile structures, and built-in heuristics to recommend tiling strategies (or "hints"). The recommended hints are easily adaptable to multiple backends, including [TVM](https://tvm.apache.org/), [triton](https://github.com/openai/triton), [tilelang](https://github.com/tile-ai/tilelang) (or other domain-specific compilers).

---

## Key Features
- **Unified Tiling Framework**: Generate tile candidates under a unified API.
- **Architecture-Specific Modeling**: Take into account Ascend NPU constraints (UB size, L1/L0 buffer sizes, CUBE unit specs).
- **Flexible Templates**: High-level templates (like `MatmulTemplate`, `GeneralReductionTemplate`, `ElementwiseTemplate`) let you concisely specify kernel structures.

---

## Usage Examples

### Basic Usage: General Reduction Template

Once installed tilelang, you can import Carver and start creating templates:

```python
from tilelang import carver
from tilelang.utils.npu_arch import AscendArch

arch = AscendArch()

# Create a general reduction template for a loop nest:
# for i in Spatial(1024):
#     for j in Spatial(1024):
#         for k in Reduce(1024):
#             ...
carve_template = carver.GeneralReductionTemplate(
    structure="SSR",
    shape=[1024, 1024, 1024],
    dtype="float16",
).with_arch(arch)

# Generate top 20 tile candidates (aka scheduling hints)
hints = carve_template.recommend_hints(topk=20)
for hint in hints:
    print(hint)
```

**Example Output** (truncated):
```python
{
  'block': [1, 128],
  'thread': [1, 128],
  'rstep': [64],
  ...
},
{
  'block': [2, 64],
  'thread': [2, 64],
  'rstep': [64],
  ...
},
...
{
  'block': [1, 16],
  'thread': [1, 16],
  'rstep': [512],
  'reduce_thread': [8],
  ...
}
```

A tile structure composed of S and R can simulate various cases. For example, structure `SS` represents a 2D element-wise operation, while `SSR` can represent a general matrix multiplication.

We can specialize more advanced templates to provide finer-grained information, such as `MatmulTemplate`.

### Matmul Template

Carver also provides a specialized `MatmulTemplate` for matrix multiplication (e.g., `C = A * B`), automatically inferring common tiling strategies (thread blocks, warps, use of tensor cores, etc.).

```python
from tilelang import carver
from tilelang.utils.npu_arch import AscendArch

arch = AscendArch()
carve_template = carver.MatmulTemplate(
    M=1024,
    N=1024,
    K=1024,
    in_dtype="float16",
    accum_dtype="float16",
    out_dtype="float16",
).with_arch(arch)

# Retrieve the (symbolic) function describing the matmul
func = carve_template.equivalent_function()
print("Equivalent Function:\n", func)

# Generate hints
hints = carve_template.recommend_hints(topk=20)
for hint in hints:
    print(hint)
```

**Example Output**:
```python
{
  'block': [32, 64],
  'warp': [16, 32],
  'rstep': [128],
  'use_tc': True,
  ...
},
{
  'block': [64, 32],
  'warp': [32, 16],
  'rstep': [128],
  'use_tc': True,
  ...
},
...
{
  'block': [256, 32],
  'warp': [128, 16],
  'rstep': [32],
  'use_tc': True,
  ...
}
```

---
## Supported Architectures

Carver of tile-ascend only provides support for:
- **NPU**

## Adapting Hints to Other Compilers

One of Carver’s main benefits is its adaptability. Here are a examples for triton lang:

Given a Carver hint like:
```python
{
  'block': [32, 64],
  'warp': [16, 32],
  'rstep': [128],
  'use_tc': True,
  'vectorize': {'A_reindex': 8, 'B_reindex': 8}
}
```
You might interpret this in **Triton** as:
- `block_m = 32, block_n = 64, block_k = 128`
- Potential warp usage = `warp_m = 16, warp_n = 32`
- `vectorize`: load data with a vector width of 8
- If `use_tc` is true, consider using Tensor Cores (TensorOps in Triton) if supported.

This helps quickly test multiple configurations without manually guessing.

## Supported Templates

Carver abstracts common loop patterns through templates:
- **`GeneralReductionTemplate`**: For general `Spatial-Spatial-Reduce` (SSR) structures or similar.
- **`MatmulTemplate`**: For standard matrix multiplication `C = A * B`.
- **`GEMVTemplate`**: For `y = Ax` or `y = xA` style operations.
- **`ElementwiseTemplate`**: For elementwise transformations or pointwise ops.

You can also create your own specialized templates if you have unique loop structures or constraints. For instance, you might define specialized templates for convolution, flash attention, etc.
