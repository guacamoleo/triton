// RUN: triton-opt %s -split-input-file -triton-amdgpu-refine-ops='arch=gfx942 granularity=small_tile' | FileCheck %s --check-prefixes=COMMON,SMALL
// RUN: triton-opt %s -split-input-file -triton-amdgpu-refine-ops='arch=gfx942 granularity=semantic_tile' | FileCheck %s --check-prefixes=COMMON,SEM

// COMMON-LABEL: @exp_kernel
// COMMON-DAG: [[VALUE_1:%.*]] = amdgpu.extract_slice {{.*}} [0, 0]
// COMMON-DAG: [[VALUE_2:%.*]] = math.exp2 [[VALUE_1]]
// COMMON-DAG: [[VALUE_3:%.*]] = amdgpu.extract_slice {{.*}} [0, 16]
// COMMON-DAG: [[VALUE_4:%.*]] = math.exp2 [[VALUE_3]]
// COMMON-DAG: [[VALUE_5:%.*]] = amdgpu.extract_slice {{.*}} [64, 0]
// COMMON-DAG: [[VALUE_6:%.*]] = math.exp2 [[VALUE_5]]
// COMMON-DAG: [[VALUE_7:%.*]] = amdgpu.extract_slice {{.*}} [64, 16]
// COMMON-DAG: [[VALUE_8:%.*]] = math.exp2 [[VALUE_7]]
// COMMON-DAG: [[VALUE_9:%.*]] = amdgpu.concat [[VALUE_2]], [[VALUE_4]], [[VALUE_6]], [[VALUE_8]]
// COMMON-DAG: tt.return [[VALUE_9]]
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @exp_kernel(%arg0: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = math.exp2 %arg0 : tensor<128x32xf32, #blocked>
    tt.return %0 : tensor<128x32xf32, #blocked>
  }
}

// -----

// COMMON-LABEL: mul_kernel
// COMMON-DAG: [[VALUE_1:%.*]] = amdgpu.extract_slice {{.*}} [0, 0]
// COMMON-DAG: [[VALUE_2:%.*]] = amdgpu.extract_slice {{.*}} [0, 0]
// COMMON-DAG: [[VALUE_3:%.*]] = arith.mulf [[VALUE_1]], [[VALUE_2]]
// COMMON-DAG: [[VALUE_4:%.*]] = amdgpu.extract_slice {{.*}} [0, 16]
// COMMON-DAG: [[VALUE_5:%.*]] = amdgpu.extract_slice {{.*}} [0, 16]
// COMMON-DAG: [[VALUE_6:%.*]] = arith.mulf [[VALUE_4]], [[VALUE_5]]
// COMMON-DAG: [[VALUE_7:%.*]] = amdgpu.extract_slice {{.*}} [64, 0]
// COMMON-DAG: [[VALUE_8:%.*]] = amdgpu.extract_slice {{.*}} [64, 0]
// COMMON-DAG: [[VALUE_9:%.*]] = arith.mulf [[VALUE_7]], [[VALUE_8]]
// COMMON-DAG: [[VALUE_10:%.*]] = amdgpu.extract_slice {{.*}} [64, 16]
// COMMON-DAG: [[VALUE_11:%.*]] = amdgpu.extract_slice {{.*}} [64, 16]
// COMMON-DAG: [[VALUE_12:%.*]] = arith.mulf [[VALUE_10]], [[VALUE_11]]
// COMMON-DAG: [[VALUE_13:%.*]] = amdgpu.concat [[VALUE_3]], [[VALUE_6]], [[VALUE_9]], [[VALUE_12]]
// COMMON-DAG: tt.return [[VALUE_13]]
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mul_kernel(%arg0: tensor<128x32xf32, #blocked>, %arg1: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = arith.mulf %arg0, %arg1 : tensor<128x32xf32, #blocked>
    tt.return %0 : tensor<128x32xf32, #blocked>
  }
}

// -----

// COMMON-LABEL: @multiple_operations_kernel

// SMALL-COUNT-4: amdgpu.extract_slice {{.*}}
// SMALL: [[OP1:%.*]] = amdgpu.concat
// SMALL-COUNT-4: amdgpu.extract_slice [[OP1]]
// SMALL: [[OP2:%.*]] = amdgpu.concat
// SMALL-COUNT-4: amdgpu.extract_slice [[OP2]]
// SMALL: [[OP3:%.*]] = amdgpu.concat
// SMALL: tt.return [[OP3]]
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @multiple_operations_kernel(%arg0: tensor<128x32xf32, #mma>, %arg1: tensor<128x32xf32, #mma>) -> tensor<128x32xf32, #mma> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = math.exp2 %arg0 : tensor<128x32xf32, #mma>
    %1 = math.exp2 %0 : tensor<128x32xf32, #mma>
    %2 = math.exp2 %1 : tensor<128x32xf32, #mma>
    tt.return %2 : tensor<128x32xf32, #mma>
  }
}

// -----

// COMMON-LABEL: @nested_operations_kernel
// COMMON-COUNT-8: amdgpu.extract_slice
// COMMON: mulf
// COMMON: amdgpu.concat
// COMMON: scf.for
// COMMON-COUNT-4: amdgpu.extract_slice
// COMMON: math.exp2
// COMMON: amdgpu.concat
// COMMON: }
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @nested_operations_kernel(%arg0: tensor<128x32xf32, #blocked>, %arg1: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = arith.mulf %arg0, %arg1 : tensor<128x32xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<128x32xf32, #blocked>) : i32 {
      %2 = math.exp2 %0 : tensor<128x32xf32, #blocked>
      scf.yield %2 : tensor<128x32xf32, #blocked>
    }
    tt.return %1 : tensor<128x32xf32, #blocked>
  }
}

// -----

// COMMON-LABEL: @peer_operations_kernel
// COMMON: scf.for
// COMMON-COUNT-4: amdgpu.extract_slice
// COMMON: math.exp2
// COMMON: amdgpu.concat
// COMMON: scf.for
// COMMON-NOT: amdgpu.extract_slice
// COMMON: math.exp2
// COMMON-NOT: amdgpu.concat
// COMMON: }
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @peer_operations_kernel(%arg0: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %arg0) -> (tensor<128x32xf32, #blocked>) : i32 {
      amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
      %2 = math.exp2 %arg2 : tensor<128x32xf32, #blocked>
      scf.yield %2 : tensor<128x32xf32, #blocked>
    }
    %3 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<128x32xf32, #blocked>) : i32 {
      %4 = math.exp2 %arg4 : tensor<128x32xf32, #blocked>
      scf.yield %4 : tensor<128x32xf32, #blocked>
    }
    tt.return %3 : tensor<128x32xf32, #blocked>
  }
}

// -----

// SMALL-DAG: [[$MMA:#.*]] = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
// SMALL-DAG: [[$LINEAR:#.*]] = #ttg.linear
// SEM-NOT: #ttg.linear
// SEM: [[$MMA:#.*]] = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
// SEM-NOT: #ttg.linear
// COMMON-LABEL: @mma_kernel
// SMALL-NOT: amdgpu.extract_slice {{.*}} : tensor<256x32xf32, [[$MMA]]> to tensor<128x8xf32, [[$MMA]]>
// SMALL-COUNT-8: amdgpu.extract_slice {{.*}} : tensor<256x32xf32, [[$MMA]]> to tensor<128x8xf32, [[$LINEAR]]>
// SEM-COUNT-2: amdgpu.extract_slice {{.*}} : tensor<256x32xf32, [[$MMA]]> to tensor<128x32xf32, [[$MMA]]>
// COMMON: amdgpu.concat
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mma_kernel(%arg0: tensor<256x32xf32, #mma>) -> tensor<256x32xf32, #mma> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = math.exp2 %arg0 : tensor<256x32xf32, #mma>
    tt.return %0 : tensor<256x32xf32, #mma>
  }
}

// -----

// SMALL-DAG: [[$LINEAR:#.*]] = #ttg.linear
// SEM-NOT: #ttg.linear
// COMMON-DAG: [[$MMA1:#.*]] = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
// COMMON-DAG: [[$MMA2:#.*]] = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
// SEM-NOT: #ttg.linear
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16384 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @convert_layout_src_linear(%arg0: tensor<128x64xf16, #mma>) attributes {noinline = false} {
    // COMMON-LABEL: convert_layout_src_linear

    // SMALL: [[ES_0:%.*]] = amdgpu.extract_slice %arg0 [0, 0] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x16xf16, [[$LINEAR]]>
    // SMALL: [[CL_0:%.*]] = ttg.convert_layout [[ES_0]] : tensor<128x16xf16, [[$LINEAR]]> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>
    // SMALL: [[ES_1:%.*]] = amdgpu.extract_slice %arg0 [0, 16] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x16xf16, [[$LINEAR]]>
    // SMALL: [[CL_1:%.*]] = ttg.convert_layout [[ES_1]] : tensor<128x16xf16, [[$LINEAR]]> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>
    // SMALL: [[ES_2:%.*]] = amdgpu.extract_slice %arg0 [0, 32] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x16xf16, [[$LINEAR]]>
    // SMALL: [[CL_2:%.*]] = ttg.convert_layout [[ES_2]] : tensor<128x16xf16, [[$LINEAR]]> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>
    // SMALL: [[ES_3:%.*]] = amdgpu.extract_slice %arg0 [0, 48] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x16xf16, [[$LINEAR]]>
    // SMALL: [[CL_3:%.*]] = ttg.convert_layout [[ES_3]] : tensor<128x16xf16, [[$LINEAR]]> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>
    // SMALL: amdgpu.concat [[CL_0]], [[CL_1]], [[CL_2]], [[CL_3]] : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>, tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>, tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>, tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>

    // SEM: [[ES_0:%.*]] = amdgpu.extract_slice %arg0 [0, 0] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x32xf16, [[$MMA1]]>
    // SEM: [[CL_0:%.*]] = ttg.convert_layout [[ES_0]] : tensor<128x32xf16, [[$MMA1]]> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>
    // SEM: [[ES_1:%.*]] = amdgpu.extract_slice %arg0 [0, 32] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x32xf16, [[$MMA1]]>
    // SEM: [[CL_1:%.*]] = ttg.convert_layout [[ES_1]] : tensor<128x32xf16, [[$MMA1]]> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>
    // SEM: amdgpu.concat [[CL_0]], [[CL_1]] : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>, tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = [[$MMA2]], kWidth = 4}>>

    %0 = ttg.convert_layout %arg0 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    tt.return
  }
}

// -----

// SEM-NOT: #ttg.linear
// SMALL-DAG: [[$LINEAR:#.*]] = #ttg.linear
// COMMON-DAG: [[$MMA1:#.*]] = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
// COMMON-DAG: [[$MMA2:#.*]] = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
// SEM-NOT: #ttg.linear
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16384 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @convert_layout_dst_linear(%arg0: tensor<128x64xf16, #mma>) attributes {noinline = false} {
    // COMMON-LABEL: convert_layout_dst_linear

    // SMALL: [[ES_0:%.*]] = amdgpu.extract_slice %arg0 [0, 0] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x16xf16, [[$MMA1]]>
    // SMALL: [[CL_0:%.*]] = ttg.convert_layout [[ES_0]] : tensor<128x16xf16, [[$MMA1]]> -> tensor<128x16xf16, [[$LINEAR]]>
    // SMALL: [[ES_1:%.*]] = amdgpu.extract_slice %arg0 [0, 16] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x16xf16, [[$MMA1]]>
    // SMALL: [[CL_1:%.*]] = ttg.convert_layout [[ES_1]] : tensor<128x16xf16, [[$MMA1]]> -> tensor<128x16xf16, [[$LINEAR]]>
    // SMALL: [[ES_2:%.*]] = amdgpu.extract_slice %arg0 [0, 32] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x16xf16, [[$MMA1]]>
    // SMALL: [[CL_2:%.*]] = ttg.convert_layout [[ES_2]] : tensor<128x16xf16, [[$MMA1]]> -> tensor<128x16xf16, [[$LINEAR]]>
    // SMALL: [[ES_3:%.*]] = amdgpu.extract_slice %arg0 [0, 48] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x16xf16, [[$MMA1]]>
    // SMALL: [[CL_3:%.*]] = ttg.convert_layout [[ES_3]] : tensor<128x16xf16, [[$MMA1]]> -> tensor<128x16xf16, [[$LINEAR]]>
    // SMALL: amdgpu.concat [[CL_0]], [[CL_1]], [[CL_2]], [[CL_3]] : tensor<128x16xf16, [[$LINEAR]]>, tensor<128x16xf16, [[$LINEAR]]>, tensor<128x16xf16, [[$LINEAR]]>, tensor<128x16xf16, [[$LINEAR]]> -> tensor<128x64xf16, [[$MMA2]]>

    // SEM: [[ES_0:%.*]] = amdgpu.extract_slice %arg0 [0, 0] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x32xf16, [[$MMA1]]>
    // SEM: [[CL_0:%.*]] = ttg.convert_layout [[ES_0]] : tensor<128x32xf16, [[$MMA1]]> -> tensor<128x32xf16, [[$MMA2]]>
    // SEM: [[ES_1:%.*]] = amdgpu.extract_slice %arg0 [0, 32] : tensor<128x64xf16, [[$MMA1]]> to tensor<128x32xf16, [[$MMA1]]>
    // SEM: [[CL_1:%.*]] = ttg.convert_layout [[ES_1]] : tensor<128x32xf16, [[$MMA1]]> -> tensor<128x32xf16, [[$MMA2]]>
    // SEM: amdgpu.concat [[CL_0]], [[CL_1]] : tensor<128x32xf16, [[$MMA2]]>, tensor<128x32xf16, [[$MMA2]]> -> tensor<128x64xf16, [[$MMA2]]>

    %0 = ttg.convert_layout %arg0 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #mma1>
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    tt.return
  }
}

// -----

// blocked layout cta tile has size of whole tensor, no transformation should happen
// COMMON-LABEL: @convert_layout_kernel_neg
// COMMON-NOT: amdgpu.extract_slice
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @convert_layout_kernel_neg(%arg0: tensor<128x32xf32, #blocked1>) -> tensor<128x32xf32, #blocked2> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = ttg.convert_layout %arg0 : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked2>
    tt.return %0 : tensor<128x32xf32, #blocked2>
  }
}
