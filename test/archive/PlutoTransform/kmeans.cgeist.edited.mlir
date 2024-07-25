#interior = affine_set<(i, j) : (i - 1 >= 0, j - 1 >= 0,  10 - i >= 0, 10 - j >= 0) > (%i, %j)
func @pad_edges(%I : memref<10x10xf32>) -> (memref<12x12xf32) {
  %O = alloc memref<12x12xf32>
  affine.parallel (%i, %j) = (0, 0) to (12, 12) {
    %1 = affine.if #interior (%i, %j) {
      %2 = load %I[%i - 1, %j - 1] : memref<10x10xf32>
      affine.yield %2
    } else {
      %2 = arith.constant 0.0 : f32
      affine.yield %2 : f32
    }
    affine.store %1, %O[%i, %j] : memref<12x12xf32>
  }
  return %O
}