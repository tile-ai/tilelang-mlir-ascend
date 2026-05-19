# 引言

本文档为​**Tilelang-AscendNPUIR**的**Autotune**模块使用指南，该模块包含了自动分块参数生成器和运行测试调优器，能够自动/半自动地求解算子所需的较优的分块参数，从而优化算子的性能。

# Autotune base

## 运行流程

1、使用Tile语言实现目标程序，并保留优化参数
2、手动或使用Carver模块生成候选配置
3、并行编译并基准测试这些候选配置，从而确定性能最佳的配置

## 使用方法

以`testing/autotune/example_gemm_autotune.py`为例，首先用户需要使用NPU版本的Tile语言实现该算子，保留对应的分块参数以供优化。
另外需要指定`out_idx`为输出tensor在prim_func中的下标位置。

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # ...existing code...

    return main
```

编写autotune需要的参考函数与数据生成函数如下：
`get_config`指定需要搜索的分块参数范围；
`ref_prog`指定torch实现的参考函数验证算子的正确性；
`supply_prog`指定算子的初始化输入张量，可省略`config`参数。

```python
# config method 1: directly defining search space in get_config function
   def get_config():
       return [
           {"block_M": 128, "block_N": 128, "K_L1": 64},
           {"block_M": 256, "block_N": 128, "K_L1": 64},
           {"block_M": 128, "block_N": 256, "K_L1": 64},
       ]

# config method 2: using itertools to generate combinations
def get_config_combination():
    block_M_options = [64, 128, 256]
    block_N_options = [64, 128, 256]
    K_L1_options = [64, 128]

    _config = list(itertools.product(block_M_options, block_N_options, K_L1_options))
    config = [{"block_M": c[0], "block_N": c[1], "K_L1": c[2]} for c in _config]
    return config

def ref_prog(A, B):
    return A @ B

def supply_prog(params, config):
    torch.manual_seed(0)
    return [
        torch.randn(M, K).half().npu(),
        torch.randn(K, N).half().npu(),
	# when use workspace with config:
	# torch.randn(M, config["K_L1"]).half().npu(),
    ]   

```

调用`@tilelang.autotune`修饰算子：

```python
@tilelang.autotune(
    configs=get_config(), # get_config_combination is also ok
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
```

指定算子并输出对应的调优分块参数：

```python
func = matmul(M, N, K)
print("Best Config:", func.get_tuner_result())
print("Test passed!")
```

# Carver

Carver是autotune模块的分块参数生成器，能够自动地生成`get_config`中的搜索范围，目前提供了三种使用方式。

## Carver Template

使用四种推荐的**Template**以进行对应的分块参数生成，在初始化**Template**时填写所需的shape参数。
目前支持的**Template**：
`MatmulTemplate`：标准矩阵乘操作
`GEMVTemplate`：y=Ax或y=xA形式的操作
`GeneralReductionTemplate`：带一个归约轴的结构
`ElementwiseTemplate`：逐元素操作

在`testing/autotune/`中提供了相应的参考代码，使用`custom_mem_mul`以自定义**Template**计算时的默认空间倍率。
例如：
在调用`ElementwiseTemplate`时，默认使用二元张量运算计算占用的ub空间大小，添加`custom_mem_mul=0.5`以得到更优的分块参数生成。
在调用`GeneralReductionTemplate`时，由于`reduce_sum`需要申请额外的ub空间，添加`custom_mem_mul=1.3`以避免生成超出ub限制的分块参数。

```python
from tilelang import carver
from tilelang.utils import AscendArch

def get_config() -> list[dict]:
    arch = AscendArch()
    carver_template = carver.MatmulTemplate(
        M = M,
        N = N,
        K = K,
        in_dtype="float16",
        accum_dtype="float16",
        out_dtype="float16",
	custom_mem_mul = 1,
    ).with_arch(arch)

    hints = carver_template.recommend_hints(topk=20)
    configs = []
    for hint in hints:
        print(hint)
        config = {
            "block_M": hint.block[0],
            "block_N": hint.block[1],
            "K_L1": hint.rstep[0],
        }
        configs.append(config)

    return configs

```

## Custom anneal carver

退火搜索模块，支持调用模板或者自定义ub、l1的空间限制与静态策略，退火搜索目前只查询16的倍数或者对应维度上的搜索上界的因子。

### 调用模板

目前支持了`FlashAttention`、`Matmul`、`Elementwise`、`Custom`四个模板

#### 使用参考

目前超参数`AnnealParam`如下，

| 参数 | 默认值  | 含义 |
|---|------|---|
| `expand_neighbour` | 2    | 退火后邻域扩展时探索的 ±步数（±1、±2） |
| `onestep_chance` | 0.7  | ±步移动 vs. 均匀随机跳转的概率 |
| `topk` | 10   | 最终返回的配置数量 |
| `number` | 100  | 初始随机种群大小 |
| `steps` | 60   | 退火迭代轮次 |
| `temperature` | 100  | 初始温度 |
| `alpha` | 0.97 | 温度衰减率 |
| `custom_mem_multiply` | 0.7  | 内存约束安全系数（乘法形式） |

`AnnealTemplate`初始化支持的自定义参数如下，

| 参数 | 默认值  | 含义 |
|---|------|---|
| `shape` | 需要给定    | 搜索算子的形状，FlashAttention为[seq_len,seq_kv_len,dim]，Elementwise任意形状 |
| `init_low_tile` | [16]*len(shape)    | 搜索下界 |
| `init_tile` | shape  | 搜索上界 |
| `only_factor` | [True]*len(shape)   | 每一维上是否使用shape的因子 |
| `annealparam` | 超参数  | 超参数 |
| `dtype` | "float16"  | 算子初始tensor数据类型 |
| `accum_dtype` | "float32"  | 算子中间tensor数据类型 |

```python
from tilelang.carver.anneal.policy import AnnealTemplate, Annealparam

def get_config():
    annealparam = Annealparam(topk=40)
    anneal_template = AnnealTemplate(shape=[seq_len, seq_len, dim], annealparam=annealparam, use_template="FlashAttention",)

    hints = anneal_template.get_configs()

    configs = []
    hint_value_min = -1
    for hint in hints:
        print(hint.kwargs)
        print(hint.value)
        configs.append({
            "block_m":hint.kwargs[0],
            "block_n":hint.kwargs[1],
            "block_k":hint.kwargs[2],
        })
    return configs
```

### Custom

搜索得到的分块切分会被作为`tile`传入约束集中，计算`Value`以进行静态时间策略计算和满足空间限制。
多项式约束：$\mathrm{Value} = \prod_i \ \mathrm{tile[i]}^\mathrm{polynomial[i]} \cdot \prod_j \ \mathrm{other \_ value[j]}$

```python
class PolyConstraints():
    polynomial: List[float]
    other_value: List[float]
```

函数约束：$\mathrm{Value} = \mathrm{func(tile[using\_tile[0]], ... \ , tile[using\_tile[len-1]]\ ,kwargs)}\cdot \prod_j \ \mathrm{other \_ value[j]}$

```python
class FuncConstraints():
    using_tile: List[int]
    other_value: List[float]
    func : Callable[..., Any] = None
    kwargs : dict
```

约束集：$\mathrm{Value} = (\sum_i \mathrm{poly\_add[i]}+\sum_j \mathrm{func\_add[j]}) \cdot \prod_k \ \mathrm{poly \_ mul[k]} \cdot \prod_p \ \mathrm{func \_ mul[p]}$

```python
class PolyConstraintsSet():
    poly_add: List[PolyConstraints]
    poly_mul: List[PolyConstraints]
    func_add: List[FuncConstraints]
    func_mul: List[FuncConstraints]
```

### 使用参考

```python
from tilelang.anneal.policy import AnnealTemplate

def get_config() -> list[dict]:
    numel_in_dtype = get_byte_per_numel("float16")
    numel_accum_dtype = get_byte_per_numel("float32") 
    seq_kv_len = seq_len
    hidden_dim = dim

    custom_kernel = CustomTemplate(
        init_low_tile = [hidden_dim, hidden_dim, hidden_dim], # search lower bound
        init_tile = [seq_len, hidden_dim, hidden_dim], # search upper bound
        only_factor = [False, True, True], # only search factors or not
        mem_caps = [PolyConstraintsSet([PolyConstraints(polynomial=[1, 0, 1], other_value=[numel_in_dtype]),
                                        PolyConstraints(polynomial=[0, 1, 1], other_value=[numel_in_dtype]),],
                                       [],[],[]), #l1_cap_1
                    PolyConstraintsSet([PolyConstraints(polynomial=[1, 1, 0], other_value=[numel_accum_dtype]),
                                        PolyConstraints(polynomial=[0, 1, 1], other_value=[numel_in_dtype]),],
                                       [],[],[]), #l1_cap_2
                    PolyConstraintsSet([PolyConstraints(polynomial=[1, 0, 1], other_value=[numel_in_dtype + 2 * numel_accum_dtype]),
                                        PolyConstraints(polynomial=[1, 1, 0], other_value=[numel_in_dtype + numel_accum_dtype]),
                                        ],
                                       [PolyConstraints(polynomial=[0, 0, 0], other_value=[0.5])],#multi_buffer numstage=2
                                       [],[]),], #ub_cap
    )
    policy = [ # time compute policy
        PolyConstraintsSet(
            [],
            [],
            [FuncConstraints([0, 1, 2,], [], _compute_carver_policy_matmul, nbytes=[numel_in_dtype, numel_in_dtype, numel_accum_dtype], init_idx=[seq_len, seq_kv_len, hidden_dim]),
             FuncConstraints([0, 2, 1,], [], _compute_carver_policy_matmul, nbytes=[numel_in_dtype, numel_in_dtype, numel_accum_dtype], init_idx=[seq_len, hidden_dim, seq_kv_len]),],
            [],
        ),#matmul 1
    ]

    mem_caps = ["L1", "L1", "UB"] # arch memory cap match custom_kernel.mem_caps
    anneal_template = AnnealTemplate(
        custom_kernel=custom_kernel,
        policy=policy,
        shape=[seq_len, seq_len, dim],
        annealparam=annealparam,
        use_template="Custom",
        mem_caps=mem_caps,
    )

    configs = []
    for hint in hints:
        print(hint.kwargs)
        print(hint.value)
        configs.append({
            "block_m":hint.kwargs[0],
            "block_n":hint.kwargs[1],
            "block_k":hint.kwargs[2],
        })
    return configs
```