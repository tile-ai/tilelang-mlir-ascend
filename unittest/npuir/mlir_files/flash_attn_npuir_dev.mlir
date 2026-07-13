module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @flash_attention(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: memref<?xf16>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.part_of_mix, mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32
    %false = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant 0.0883883461 : f32
    %cst_1 = arith.constant 0xFF800000 : f32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [512, 128], strides: [%c128, %c1] : memref<?xf16> to memref<512x128xf16, strided<[128, 1]>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512, 128], strides: [%c128, %c1] : memref<?xf16> to memref<512x128xf16, strided<[128, 1]>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [512, 128], strides: [%c128, %c1] : memref<?xf16> to memref<512x128xf16, strided<[128, 1]>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [512, 128], strides: [%c128, %c1] : memref<?xf16> to memref<512x128xf16, strided<[128, 1]>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = tensor.empty() : tensor<64x1xf32>
    %3 = tensor.empty() : tensor<64x1xf32>
    %4 = tensor.empty() : tensor<64x128xf32>
    %5 = tensor.empty() : tensor<64x64xf32>
    %6 = tensor.empty() : tensor<64x128xf16>
    %7 = arith.muli %1, %c64_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %subview = memref.subview %reinterpret_cast[%8, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>>
    %alloc = memref.alloc() : memref<64x128xf16>
    memref.copy %subview, %alloc : memref<64x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16>
    %9 = bufferization.to_tensor %alloc restrict : memref<64x128xf16>
    %10 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%3 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %12 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %13 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %14:3 = scf.for %arg12 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg13 = %12, %arg14 = %11, %arg15 = %10) -> (tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x128xf32>)  : i32 {
      %17 = tensor.empty() : tensor<64x64xf32>
      %18 = tensor.empty() : tensor<64x64xf16>
      %19 = tensor.empty() : tensor<64x1xf32>
      %20 = tensor.empty() : tensor<64x1xf32>
      %21 = tensor.empty() : tensor<64x1xf32>
      %22 = tensor.empty() : tensor<64x64xf32>
      %23 = tensor.empty() : tensor<64x1xf32>
      %24 = tensor.empty() : tensor<64x1xf32>
      %25 = arith.muli %arg12, %c64_i32 : i32
      %26 = arith.index_cast %25 : i32 to index
      %subview_6 = memref.subview %reinterpret_cast_3[%26, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>>
      %alloc_7 = memref.alloc() : memref<64x128xf16>
      memref.copy %subview_6, %alloc_7 : memref<64x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16>
      %27 = bufferization.to_tensor %alloc_7 restrict : memref<64x128xf16>
      %28 = hivm.hir.mmadL1 {b_transpose} ins(%9, %27, %true, %c64, %c64, %c128 : tensor<64x128xf16>, tensor<64x128xf16>, i1, index, index, index) outs(%17 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %29 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%28, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%28 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %collapsed_8 = tensor.collapse_shape %20 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
      %reduced = linalg.reduce ins(%29 : tensor<64x64xf32>) outs(%collapsed_8 : tensor<64xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %43 = arith.maximumf %in, %init : f32
          linalg.yield %43 : f32
        }
      %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %30 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%arg13, %expanded : tensor<64x1xf32>, tensor<64x1xf32>) outs(%24 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %31 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg13, %30 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%23 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %32 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%31 : tensor<64x1xf32>) outs(%19 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %collapsed_9 = tensor.collapse_shape %30 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
      %broadcasted_10 = linalg.broadcast ins(%collapsed_9 : tensor<64xf32>) outs(%22 : tensor<64x64xf32>) dimensions = [1]
      %33 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%29, %broadcasted_10 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%22 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %34 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%33 : tensor<64x64xf32>) outs(%29 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %collapsed_11 = tensor.collapse_shape %21 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
      %reduced_12 = linalg.reduce ins(%34 : tensor<64x64xf32>) outs(%collapsed_11 : tensor<64xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %43 = arith.addf %in, %init : f32
          linalg.yield %43 : f32
        }
      %expanded_13 = tensor.expand_shape %reduced_12 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %35 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg14, %32 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%arg14 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %36 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%35, %expanded_13 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%35 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %collapsed_14 = tensor.collapse_shape %32 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
      %broadcasted_15 = linalg.broadcast ins(%collapsed_14 : tensor<64xf32>) outs(%arg15 : tensor<64x128xf32>) dimensions = [1]
      %37 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg15, %broadcasted_15 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%arg15 : tensor<64x128xf32>) -> tensor<64x128xf32>
      %38 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>, type_fn = #hfusion.type_fn<cast_signed>} ins(%34 : tensor<64x64xf32>) outs(%18 : tensor<64x64xf16>) -> tensor<64x64xf16>
      %39 = linalg.fill ins(%cst : f32) outs(%31 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %40 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%39, %30 : tensor<64x1xf32>, tensor<64x1xf32>) outs(%arg13 : tensor<64x1xf32>) -> tensor<64x1xf32>
      %subview_16 = memref.subview %reinterpret_cast_2[%26, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>>
      %alloc_17 = memref.alloc() : memref<64x128xf16>
      memref.copy %subview_16, %alloc_17 : memref<64x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16>
      %41 = bufferization.to_tensor %alloc_17 restrict : memref<64x128xf16>
      %42 = hivm.hir.mmadL1 ins(%38, %41, %false, %c64, %c64, %c128 : tensor<64x64xf16>, tensor<64x128xf16>, i1, index, index, index) outs(%37 : tensor<64x128xf32>) -> tensor<64x128xf32>
      scf.yield %40, %36, %42 : tensor<64x1xf32>, tensor<64x1xf32>, tensor<64x128xf32>
    }
    %collapsed = tensor.collapse_shape %14#1 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%14#2 : tensor<64x128xf32>) dimensions = [1]
    %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%14#2, %broadcasted : tensor<64x128xf32>, tensor<64x128xf32>) outs(%14#2 : tensor<64x128xf32>) -> tensor<64x128xf32>
    %16 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>, type_fn = #hfusion.type_fn<cast_signed>} ins(%15 : tensor<64x128xf32>) outs(%6 : tensor<64x128xf16>) -> tensor<64x128xf16>
    %subview_5 = memref.subview %reinterpret_cast_4[%8, 0] [64, 128] [1, 1] : memref<512x128xf16, strided<[128, 1]>> to memref<64x128xf16, strided<[128, 1], offset: ?>>
    bufferization.materialize_in_destination %16 in writable %subview_5 : (tensor<64x128xf16>, memref<64x128xf16, strided<[128, 1], offset: ?>>) -> ()
    return
  }
}
