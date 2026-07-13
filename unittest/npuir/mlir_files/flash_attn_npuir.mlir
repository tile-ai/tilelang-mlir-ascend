module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @FlashAttnExp_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: memref<?xf16, #hivm.address_space<gm>>, %arg8: memref<?xf16, #hivm.address_space<gm>>, %arg9: memref<?xf16, #hivm.address_space<gm>>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c256 = arith.constant 256 : index
    %c4096 = arith.constant 4096 : index
    %c2048 = arith.constant 2048 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c256_i32 = arith.constant 256 : i32
    %c96_i32 = arith.constant 96 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4096, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<4096x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [4096, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<4096x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4096, 2048], strides: [%c2048, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<4096x2048xf16, strided<[2048, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [4096, 4096], strides: [%c4096, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<4096x4096xf16, strided<[4096, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4096, 4096], strides: [%c4096, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<4096x4096xf16, strided<[4096, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4096, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<4096x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %alloc = memref.alloc() : memref<96x256xf16, #hivm.address_space<cbuf>>
    %alloc_5 = memref.alloc() : memref<256x128xf16, #hivm.address_space<cbuf>>
    %alloc_6 = memref.alloc() : memref<96x256xf32, #hivm.address_space<cc>>
    scf.for %arg16 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      %2 = arith.muli %1, %c96_i32 : i32
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.subi %c4096_i32, %2 : i32
      %5 = arith.minsi %4, %c96_i32 : i32
      %6 = arith.index_cast %5 : i32 to index
      %subview = memref.subview %reinterpret_cast[%3, 0] [%6, 128] [1, 1] : memref<4096x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %alloc[0, 0] [%6, 128] [1, 1] : memref<96x256xf16, #hivm.address_space<cbuf>> to memref<?x128xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?x128xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>)
      %7 = arith.muli %arg16, %c256_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %subview_8 = memref.subview %reinterpret_cast_4[%8, 0] [256, 128] [1, 1] : memref<4096x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview_8 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_5 : memref<256x128xf16, #hivm.address_space<cbuf>>)
      hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_5, %true, %6, %c256, %c128 : memref<96x256xf16, #hivm.address_space<cbuf>>, memref<256x128xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_6 : memref<96x256xf32, #hivm.address_space<cc>>)
      %subview_9 = memref.subview %alloc_6[0, 0] [%6, 256] [1, 1] : memref<96x256xf32, #hivm.address_space<cc>> to memref<?x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>
      %subview_10 = memref.subview %reinterpret_cast_2[%3, %8] [%6, 256] [1, 1] : memref<4096x4096xf16, strided<[4096, 1]>, #hivm.address_space<gm>> to memref<?x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.fixpipe ins(%subview_9 : memref<?x256xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%subview_10 : memref<?x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>)
      %9 = arith.extsi %arg16 : i32 to i64
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %9
    }
    scf.for %arg16 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      %2 = arith.extsi %arg16 : i32 to i64
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_MTE2>] flag = %2
      %3 = arith.muli %1, %c96_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.subi %c4096_i32, %3 : i32
      %6 = arith.minsi %5, %c96_i32 : i32
      %7 = arith.index_cast %6 : i32 to index
      %8 = arith.muli %arg16, %c256_i32 : i32
      %9 = arith.index_cast %8 : i32 to index
      %subview = memref.subview %reinterpret_cast_3[%4, %9] [%7, 256] [1, 1] : memref<4096x4096xf16, strided<[4096, 1]>, #hivm.address_space<gm>> to memref<?x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %alloc[0, 0] [%7, 256] [1, 1] : memref<96x256xf16, #hivm.address_space<cbuf>> to memref<?x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?x256xf16, strided<[256, 1]>, #hivm.address_space<cbuf>>)
      %subview_8 = memref.subview %reinterpret_cast_0[%9, 0] [256, 128] [1, 1] : memref<4096x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.nd2nz {dst_continuous} ins(%subview_8 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_5 : memref<256x128xf16, #hivm.address_space<cbuf>>)
      hivm.hir.mmadL1 ins(%alloc, %alloc_5, %true, %7, %c256, %c128 : memref<96x256xf16, #hivm.address_space<cbuf>>, memref<256x128xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_6 : memref<96x256xf32, #hivm.address_space<cc>>)
      %subview_9 = memref.subview %alloc_6[0, 0] [%7, 128] [1, 1] : memref<96x256xf32, #hivm.address_space<cc>> to memref<?x128xf32, strided<[256, 1]>, #hivm.address_space<cc>>
      %10 = arith.muli %arg16, %c128_i32 : i32
      %11 = arith.index_cast %10 : i32 to index
      %subview_10 = memref.subview %reinterpret_cast_1[%4, %11] [%7, 128] [1, 1] : memref<4096x2048xf16, strided<[2048, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.fixpipe ins(%subview_9 : memref<?x128xf32, strided<[256, 1]>, #hivm.address_space<cc>>) outs(%subview_10 : memref<?x128xf16, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>)
      %12 = arith.extsi %arg16 : i32 to i64
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = %12
    }
    return
  }
  func.func @FlashAttnExp_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xf16, #hivm.address_space<gm>>, %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: memref<?xf16, #hivm.address_space<gm>>, %arg8: memref<?xf16, #hivm.address_space<gm>>, %arg9: memref<?xf16, #hivm.address_space<gm>>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c4096 = arith.constant 4096 : index
    %c2048 = arith.constant 2048 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c48_i32 = arith.constant 48 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %c256_i32 = arith.constant 256 : i32
    %c2_i32 = arith.constant 2 : i32
    %c96_i32 = arith.constant 96 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c0_i32 = arith.constant 0 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [0], sizes: [4096, 2048], strides: [%c2048, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<4096x2048xf16, strided<[2048, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [4096, 4096], strides: [%c4096, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<4096x4096xf16, strided<[4096, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [4096, 4096], strides: [%c4096, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<4096x4096xf16, strided<[4096, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [4096, 128], strides: [%c128, %c1] : memref<?xf16, #hivm.address_space<gm>> to memref<4096x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = hivm.hir.get_sub_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %alloc = memref.alloc() : memref<48x1xf32, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() : memref<48x1xf32, #hivm.address_space<ub>>
    %alloc_6 = memref.alloc() : memref<48x1xf32, #hivm.address_space<ub>>
    %alloc_7 = memref.alloc() : memref<768x1xf32, #hivm.address_space<ub>>
    %alloc_8 = memref.alloc() : memref<48x128xf16, #hivm.address_space<ub>>
    %alloc_9 = memref.alloc() : memref<48x128xf32, #hivm.address_space<ub>>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<48x1xf32, #hivm.address_space<ub>>)
    linalg.fill ins(%cst : f32) outs(%alloc_9 : memref<48x128xf32, #hivm.address_space<ub>>)
    linalg.fill ins(%cst : f32) outs(%alloc_6 : memref<48x1xf32, #hivm.address_space<ub>>)
    linalg.fill ins(%cst : f32) outs(%alloc_7 : memref<768x1xf32, #hivm.address_space<ub>>)
    linalg.fill ins(%cst_1 : f32) outs(%alloc_5 : memref<48x1xf32, #hivm.address_space<ub>>)
    scf.for %arg16 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      %alloc_12 = memref.alloc() : memref<48x1xf32, #hivm.address_space<ub>>
      %alloc_13 = memref.alloc() : memref<48x1xf32, #hivm.address_space<ub>>
      %alloc_14 = memref.alloc() : memref<48x256xf16, #hivm.address_space<ub>>
      %alloc_15 = memref.alloc() : memref<48x256xf32, #hivm.address_space<ub>>
      memref.copy %alloc_5, %alloc_12 : memref<48x1xf32, #hivm.address_space<ub>> to memref<48x1xf32, #hivm.address_space<ub>>
      %13 = arith.extsi %arg16 : i32 to i64
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %13
      %14 = arith.muli %1, %c96_i32 : i32
      %15 = arith.subi %c4096_i32, %14 : i32
      %16 = arith.minsi %15, %c96_i32 : i32
      %17 = arith.addi %16, %c1_i32 : i32
      %18 = arith.divsi %17, %c2_i32 : i32
      %19 = arith.muli %3, %18 : i32
      %20 = arith.addi %14, %19 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %18 : i32 to index
      %23 = arith.muli %arg16, %c256_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview_16 = memref.subview %reinterpret_cast_2[%21, %24] [%22, 256] [1, 1] : memref<4096x4096xf16, strided<[4096, 1]>, #hivm.address_space<gm>> to memref<?x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_17 = memref.subview %alloc_14[0, 0] [%22, 256] [1, 1] : memref<48x256xf16, #hivm.address_space<ub>> to memref<?x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
      memref.copy %subview_16, %subview_17 : memref<?x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
      hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>, type_fn = #hfusion.type_fn<cast_signed>} ins(%alloc_14 : memref<48x256xf16, #hivm.address_space<ub>>) outs(%alloc_15 : memref<48x256xf32, #hivm.address_space<ub>>)
      linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%alloc_15, %cst_0 : memref<48x256xf32, #hivm.address_space<ub>>, f32) outs(%alloc_15 : memref<48x256xf32, #hivm.address_space<ub>>)
      %collapse_shape = memref.collapse_shape %alloc_5 [[0, 1]] : memref<48x1xf32, #hivm.address_space<ub>> into memref<48xf32, #hivm.address_space<ub>>
      linalg.reduce ins(%alloc_15 : memref<48x256xf32, #hivm.address_space<ub>>) outs(%collapse_shape : memref<48xf32, #hivm.address_space<ub>>) dimensions = [1]
        (%in: f32, %init: f32) {
          %27 = arith.maximumf %in, %init : f32
          linalg.yield %27 : f32
        }
      %25 = arith.cmpi sgt, %arg16, %c0_i32 : i32
      scf.if %25 {
        hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%alloc_12, %alloc_5 : memref<48x1xf32, #hivm.address_space<ub>>, memref<48x1xf32, #hivm.address_space<ub>>) outs(%alloc_5 : memref<48x1xf32, #hivm.address_space<ub>>)
        linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%alloc_12, %alloc_5 : memref<48x1xf32, #hivm.address_space<ub>>, memref<48x1xf32, #hivm.address_space<ub>>) outs(%alloc_6 : memref<48x1xf32, #hivm.address_space<ub>>)
        linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%alloc_6 : memref<48x1xf32, #hivm.address_space<ub>>) outs(%alloc_6 : memref<48x1xf32, #hivm.address_space<ub>>)
        %27 = arith.muli %arg16, %c48_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %subview_22 = memref.subview %alloc_7[%28, 0] [48, 1] [1, 1] : memref<768x1xf32, #hivm.address_space<ub>> to memref<48x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        memref.copy %alloc_6, %subview_22 : memref<48x1xf32, #hivm.address_space<ub>> to memref<48x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      }
      %alloc_18 = memref.alloc() : memref<48x256xf32, #hivm.address_space<ub>>
      %collapse_shape_18 = memref.collapse_shape %alloc_5 [[0, 1]] : memref<48x1xf32, #hivm.address_space<ub>> into memref<48xf32, #hivm.address_space<ub>>
      linalg.broadcast ins(%collapse_shape_18 : memref<48xf32, #hivm.address_space<ub>>) outs(%alloc_18 : memref<48x256xf32, #hivm.address_space<ub>>) dimensions = [1]
      linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%alloc_15, %alloc_18 : memref<48x256xf32, #hivm.address_space<ub>>, memref<48x256xf32, #hivm.address_space<ub>>) outs(%alloc_15 : memref<48x256xf32, #hivm.address_space<ub>>)
      linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%alloc_15 : memref<48x256xf32, #hivm.address_space<ub>>) outs(%alloc_15 : memref<48x256xf32, #hivm.address_space<ub>>)
      hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>, type_fn = #hfusion.type_fn<cast_signed>} ins(%alloc_15 : memref<48x256xf32, #hivm.address_space<ub>>) outs(%alloc_14 : memref<48x256xf16, #hivm.address_space<ub>>)
      %subview_19 = memref.subview %alloc_14[0, 0] [%22, 256] [1, 1] : memref<48x256xf16, #hivm.address_space<ub>> to memref<?x256xf16, strided<[256, 1]>, #hivm.address_space<ub>>
      %subview_20 = memref.subview %reinterpret_cast_3[%21, %24] [%22, 256] [1, 1] : memref<4096x4096xf16, strided<[4096, 1]>, #hivm.address_space<gm>> to memref<?x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
      memref.copy %subview_19, %subview_20 : memref<?x256xf16, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x256xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
      %26 = arith.extsi %arg16 : i32 to i64
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = %26
      %collapse_shape_21 = memref.collapse_shape %alloc_13 [[0, 1]] : memref<48x1xf32, #hivm.address_space<ub>> into memref<48xf32, #hivm.address_space<ub>>
      linalg.reduce ins(%alloc_15 : memref<48x256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<48xf32, #hivm.address_space<ub>>) dimensions = [1]
        (%in: f32, %init: f32) {
          %27 = arith.addf %in, %init : f32
          linalg.yield %27 : f32
        }
      linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%alloc, %alloc_6 : memref<48x1xf32, #hivm.address_space<ub>>, memref<48x1xf32, #hivm.address_space<ub>>) outs(%alloc : memref<48x1xf32, #hivm.address_space<ub>>)
      linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%alloc, %alloc_13 : memref<48x1xf32, #hivm.address_space<ub>>, memref<48x1xf32, #hivm.address_space<ub>>) outs(%alloc : memref<48x1xf32, #hivm.address_space<ub>>)
    }
    scf.for %arg16 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      %alloc_12 = memref.alloc() : memref<48x128xf32, #hivm.address_space<ub>>
      %13 = arith.extsi %arg16 : i32 to i64
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_MTE2>] flag = %13
      %14 = arith.muli %1, %c96_i32 : i32
      %15 = arith.subi %c4096_i32, %14 : i32
      %16 = arith.minsi %15, %c96_i32 : i32
      %17 = arith.addi %16, %c1_i32 : i32
      %18 = arith.divsi %17, %c2_i32 : i32
      %19 = arith.muli %3, %18 : i32
      %20 = arith.addi %14, %19 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %18 : i32 to index
      %23 = arith.muli %arg16, %c128_i32 : i32
      %24 = arith.index_cast %23 : i32 to index
      %subview_13 = memref.subview %reinterpret_cast[%21, %24] [%22, 128] [1, 1] : memref<4096x2048xf16, strided<[2048, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_14 = memref.subview %alloc_8[0, 0] [%22, 128] [1, 1] : memref<48x128xf16, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      memref.copy %subview_13, %subview_14 : memref<?x128xf16, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
      hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>, type_fn = #hfusion.type_fn<cast_signed>} ins(%alloc_8 : memref<48x128xf16, #hivm.address_space<ub>>) outs(%alloc_12 : memref<48x128xf32, #hivm.address_space<ub>>)
      %25 = arith.cmpi sgt, %arg16, %c0_i32 : i32
      scf.if %25 {
        %26 = arith.muli %arg16, %c48_i32 : i32
        %27 = arith.index_cast %26 : i32 to index
        %subview_16 = memref.subview %alloc_7[%27, 0] [48, 1] [1, 1] : memref<768x1xf32, #hivm.address_space<ub>> to memref<48x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
        memref.copy %subview_16, %alloc_6 : memref<48x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>> to memref<48x1xf32, #hivm.address_space<ub>>
      }
      %alloc_15 = memref.alloc() : memref<48x128xf32, #hivm.address_space<ub>>
      %collapse_shape_15 = memref.collapse_shape %alloc_6 [[0, 1]] : memref<48x1xf32, #hivm.address_space<ub>> into memref<48xf32, #hivm.address_space<ub>>
      linalg.broadcast ins(%collapse_shape_15 : memref<48xf32, #hivm.address_space<ub>>) outs(%alloc_15 : memref<48x128xf32, #hivm.address_space<ub>>) dimensions = [1]
      linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%alloc_9, %alloc_15 : memref<48x128xf32, #hivm.address_space<ub>>, memref<48x128xf32, #hivm.address_space<ub>>) outs(%alloc_9 : memref<48x128xf32, #hivm.address_space<ub>>)
      linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%alloc_9, %alloc_12 : memref<48x128xf32, #hivm.address_space<ub>>, memref<48x128xf32, #hivm.address_space<ub>>) outs(%alloc_9 : memref<48x128xf32, #hivm.address_space<ub>>)
    }
    %alloc_10 = memref.alloc() : memref<48x128xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %alloc [[0, 1]] : memref<48x1xf32, #hivm.address_space<ub>> into memref<48xf32, #hivm.address_space<ub>>
    linalg.broadcast ins(%collapse_shape : memref<48xf32, #hivm.address_space<ub>>) outs(%alloc_10 : memref<48x128xf32, #hivm.address_space<ub>>) dimensions = [1]
    linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%alloc_9, %alloc_10 : memref<48x128xf32, #hivm.address_space<ub>>, memref<48x128xf32, #hivm.address_space<ub>>) outs(%alloc_9 : memref<48x128xf32, #hivm.address_space<ub>>)
    hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>, type_fn = #hfusion.type_fn<cast_signed>} ins(%alloc_9 : memref<48x128xf32, #hivm.address_space<ub>>) outs(%alloc_8 : memref<48x128xf16, #hivm.address_space<ub>>)
    %4 = arith.muli %1, %c96_i32 : i32
    %5 = arith.subi %c4096_i32, %4 : i32
    %6 = arith.minsi %5, %c96_i32 : i32
    %7 = arith.addi %6, %c1_i32 : i32
    %8 = arith.divsi %7, %c2_i32 : i32
    %9 = arith.index_cast %8 : i32 to index
    %subview = memref.subview %alloc_8[0, 0] [%9, 128] [1, 1] : memref<48x128xf16, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>>
    %10 = arith.muli %3, %8 : i32
    %11 = arith.addi %4, %10 : i32
    %12 = arith.index_cast %11 : i32 to index
    %subview_11 = memref.subview %reinterpret_cast_4[%12, 0] [%9, 128] [1, 1] : memref<4096x128xf16, strided<[128, 1]>, #hivm.address_space<gm>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %subview_11 : memref<?x128xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
