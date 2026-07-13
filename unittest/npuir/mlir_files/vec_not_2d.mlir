module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @main(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xi16, #hivm.address_space<gm>>, %arg4: memref<?xi16, #hivm.address_space<gm>>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mix_mode = "aiv"} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c-1_i16 = arith.constant -1 : i16
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    hivm.hir.set_ffts_base_addr %arg0
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [512, 512], strides: [%c512, %c1] : memref<?xi16, #hivm.address_space<gm>> to memref<512x512xi16, strided<[512, 1]>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512, 512], strides: [%c512, %c1] : memref<?xi16, #hivm.address_space<gm>> to memref<512x512xi16, strided<[512, 1]>, #hivm.address_space<gm>>
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %alloc = memref.alloc() : memref<128x256xi16, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<128x256xi16, #hivm.address_space<ub>>
    %2 = arith.divsi %1, %c2_i32 : i32
    %3 = arith.muli %2, %c128_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = arith.remsi %1, %c2_i32 : i32
    %6 = arith.muli %5, %c256_i32 : i32
    %7 = arith.index_cast %6 : i32 to index
    %subview = memref.subview %reinterpret_cast[%4, %7] [128, 256] [1, 1] : memref<512x512xi16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x256xi16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %subview, %alloc_1 : memref<128x256xi16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>> to memref<128x256xi16, #hivm.address_space<ub>>
    hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%alloc_1 : memref<128x256xi16, #hivm.address_space<ub>>) outs(%alloc : memref<128x256xi16, #hivm.address_space<ub>>)
    %subview_2 = memref.subview %reinterpret_cast_0[%4, %7] [128, 256] [1, 1] : memref<512x512xi16, strided<[512, 1]>, #hivm.address_space<gm>> to memref<128x256xi16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    memref.copy %alloc, %subview_2 : memref<128x256xi16, #hivm.address_space<ub>> to memref<128x256xi16, strided<[512, 1], offset: ?>, #hivm.address_space<gm>>
    return
  }
}
