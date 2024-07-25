#set = affine_set<()[s0] : (s0 - 1 >= 0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func @k_classify_local(%arg0: memref<?xf32>, %arg1: i32, %arg2: i32, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xi32>, %arg6: i32, %arg7: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %arg6 : i32 to index
    %1 = arith.index_cast %arg7 : i32 to index
    %2 = llvm.mlir.undef : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %2, %alloca[] : memref<f32>
    %alloca_0 = memref.alloca() : memref<f32>
    affine.store %2, %alloca_0[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<i32>
    %3 = llvm.mlir.undef : i32
    affine.store %3, %alloca_1[] : memref<i32>
    %4 = arith.index_cast %arg1 : i32 to index
    affine.for %arg8 = 0 to %4 {
      affine.store %c-1_i32, %alloca_1[] : memref<i32>
      affine.for %arg9 = 0 to %0 {
        %9 = arith.index_cast %arg9 : index to i32
        affine.store %cst, %alloca[] : memref<f32>
        affine.if #set()[%1] {
          %16 = affine.load %arg3[0] : memref<?xf32>
          %17 = affine.load %arg0[0] : memref<?xf32>
          %18 = arith.subf %16, %17 : f32
          %19 = arith.mulf %18, %18 : f32
          affine.for %arg10 = 0 to %1 {
            %20 = affine.load %alloca[] : memref<f32>
            %21 = arith.addf %20, %19 : f32
            affine.store %21, %alloca[] : memref<f32>
          }
        }
        %10 = affine.load %alloca[] : memref<f32>
        %11 = affine.load %alloca_0[] : memref<f32>
        %12 = arith.cmpf olt, %10, %11 : f32
        %13 = arith.select %12, %10, %11 : f32
        affine.store %13, %alloca_0[] : memref<f32>
        %14 = arith.cmpf olt, %10, %13 : f32
        %15 = scf.if %14 -> (i32) {
          scf.yield %9 : i32
        } else {
          %16 = affine.load %alloca_1[] : memref<i32>
          scf.yield %16 : i32
        }
        affine.store %15, %alloca_1[] : memref<i32>
      }
      %5 = affine.load %alloca_1[] : memref<i32>
      %6 = arith.index_cast %5 : i32 to index
      %7 = memref.load %arg5[%6] : memref<?xi32>
      %8 = arith.addi %7, %c1_i32 : i32
      memref.store %8, %arg5[%6] : memref<?xi32>
      affine.for %arg9 = 0 to %1 {
        %9 = affine.load %arg0[0] : memref<?xf32>
        %10 = affine.load %arg4[0] : memref<?xf32>
        %11 = arith.addf %10, %9 : f32
        affine.store %11, %arg4[0] : memref<?xf32>
      }
    }
    return
  }
}
