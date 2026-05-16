# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import os
import torch
import tilelang
import tilelang.language as T
import torch.nn.functional as F

seq_len = 2048
dim = 128

torch.npu.set_device(10)

@tilelang.jit(target="npuir")
def online_flash_attention(block_M, block_N,multibuffer, dtype="float16", accum_dtype="float32"):
    shape_q = [seq_len, dim]
    shape_k = [seq_len, dim]
    shape_v = [seq_len, dim]
    shape_o = [seq_len, dim]

    block_m = block_M
    block_n = block_N
    ROW_BLOCKS = T.ceildiv(seq_len, block_m)
    num_n = T.ceildiv(seq_len, block_n)

    multi_buffer = multibuffer
    PHYSICAL_CORE_NUM = 24
    
    block_vid = block_m // 2
    @T.prim_func
    def flash_attention(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        V: T.Tensor(shape_v, dtype),
        Output: T.Tensor(shape_o, dtype),

    ):
        with T.Kernel(PHYSICAL_CORE_NUM, is_npu=True) as (kernel_id, vid):

            Q_shared = T.alloc_shared([block_m, dim], dtype)
            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            scores = T.alloc_fragment([block_m, block_n], accum_dtype)
            socres_l1 = T.alloc_shared([block_m, block_n], dtype)
            acc_o_l0c = T.alloc_fragment([block_m, dim], accum_dtype)
            scores_ub = T.alloc_shared([block_vid, block_n], accum_dtype) 
            scores_cast = T.alloc_shared([block_vid, block_n], dtype)
            acc_m = T.alloc_shared([block_vid, 1], accum_dtype)
            acc_l = T.alloc_shared([block_vid, 1], accum_dtype)
            acc_o_ub = T.alloc_shared([block_vid, dim], accum_dtype)
            local_max = T.alloc_shared([block_vid,1], accum_dtype)
            local_sum = T.alloc_shared([block_vid,1], accum_dtype)
            new_max = T.alloc_shared([block_vid,1], accum_dtype)
            correction = T.alloc_shared([block_vid,1], accum_dtype, multi_buffer = multi_buffer)
            tmp1 = T.alloc_shared([block_vid,1], accum_dtype)
            acc_o = T.alloc_shared([block_vid, dim], accum_dtype)
            
            Workspace1 = T.alloc_workspace((2, block_m, block_n), accum_dtype, multi_buffer = multi_buffer)
            Workspace2 = T.alloc_workspace((2, block_m, block_n), dtype, multi_buffer = multi_buffer)
            Workspace3 = T.alloc_workspace((2, block_m, dim), accum_dtype, multi_buffer = multi_buffer)

            value_zero = 0
            scale = (1.0 / dim)**0.5
            value_min = -T.infinity(accum_dtype)
            
            num_local_tasks = T.ceildiv(ROW_BLOCKS - kernel_id, PHYSICAL_CORE_NUM)
            
            for task_id in T.serial(num_local_tasks):
                cid = kernel_id + task_id * PHYSICAL_CORE_NUM

                offset = cid * block_m
                tail_size_m = T.min(block_m, seq_len - cid * block_m)
                T.copy(Q[offset, 0], Q_shared, size=[block_m, dim])

                T.vbrc(value_zero, acc_o)
                T.vbrc(value_zero, acc_l)
                T.vbrc(value_min, acc_m)
            

                for k in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=multi_buffer):
                    # cube
                    T.copy(K[k* block_n, 0], K_shared, size=[block_n, dim])
                    T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True, size=[block_m, dim, block_n])

                    T.copy(scores, Workspace1[0,0, 0], size = [block_m, block_n])
                    
                    # vec
                    T.copy(Workspace1[0,vid * block_m // 2, 0], scores_ub, size=[block_m // 2, block_n])
                    T.vmul(scores_ub, scale, scores_ub) 
                    T.reduce_max(scores_ub, local_max, dim=1)
                    T.vmax(acc_m, local_max, new_max)
                    T.vsub(scores_ub, new_max, scores_ub)
                    T.vexp(scores_ub, scores_ub)
                    T.vcast(scores_ub, scores_cast, round_mode="rint")
                    T.reduce_sum(scores_ub, local_sum, dim=1)
                    T.copy(scores_cast, Workspace2[0,vid * block_m // 2, 0], size=[block_m // 2, block_n])
                    
                    T.vsub(acc_m, new_max ,tmp1)
                    T.vexp(tmp1, correction)
                    
                    
                    T.vmul(acc_l, correction, acc_l)
                    T.vadd(acc_l, local_sum, acc_l)
                    
                    T.vbrc(value_zero, tmp1)
                    T.vadd(tmp1, new_max, acc_m)

                    
                    # cube
                    T.copy(Workspace2[0, 0, 0],  socres_l1)
                    T.copy(V[k * block_n, 0], V_shared, size=[block_n, dim])
                    T.gemm(socres_l1, V_shared, acc_o_l0c, initC=True, size=[block_m, block_n, dim])
                    T.copy(acc_o_l0c, Workspace3[0,0, 0], size=[block_m, dim])

                    
                    # vec
                    T.copy(Workspace3[0,vid * block_m // 2, 0], acc_o_ub, size=[block_m // 2, dim])

                    T.vmul(acc_o, correction, acc_o)
                    T.vadd(acc_o, acc_o_ub, acc_o)
                
                T.vdiv(acc_o, acc_l, acc_o)
                
                O_cast = T.alloc_shared([block_vid, dim], dtype)
                T.vcast(acc_o, O_cast, round_mode="rint")
                real_m = T.min(block_m // 2, seq_len - cid * block_m - vid * block_m // 2)
                T.copy(O_cast, Output[cid * block_m + vid * block_m // 2, 0], size=[real_m, dim])
            
    return flash_attention

def main():
    # In the futrue, Developer mode and Expert Mode will transition smoothly without
    # requiring explicit declarations.
    os.environ['TILELANG_ASCEND_WORKSPACE_SIZE'] = str(1024*1024*1024*8)
    bm = 128
    bn = 256
    multibuffer = 8
    kernel = online_flash_attention(bm, bn,multibuffer)
    torch.manual_seed(88888888) 
    q = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    k = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    v = torch.randn((seq_len, dim), dtype=torch.float16).npu()
    output = torch.zeros((seq_len, dim), dtype=torch.float16).npu()

    kernel(q, k, v, output)
    scale = (1.0 / dim)**0.5
    ref_output = torch.nn.functional.softmax(
        (q @ k.T).to(torch.float32) * scale, dim=-1).to(torch.float16) @ v
    
    
    print("output:",output)
    print("ref_output:",ref_output)
    torch.testing.assert_close(ref_output, output, rtol=1e-3, atol=1e-3)
    print("All check passed.")

if __name__ == "__main__":
    main()
