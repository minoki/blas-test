import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector.Unboxed.Mutable as VUM
import Control.Monad (forM_)
import Foreign.CUDA.Runtime.Marshal as CUDA
import Foreign.CUDA.BLAS as CuBLAS
import Foreign.CUDA.BLAS.Level3 as CuBLAS
import Foreign.Marshal (with)

printMat :: Int -> Int -> VU.Vector Double -> IO ()
printMat m n v
  | VU.length v == m * n
  = forM_ [0..m-1] $ \i -> do
      print $ VU.slice (n * i) n v
  | otherwise = error "size mismatch"

main :: IO ()
main = do
  handle <- CuBLAS.create
  ptrA <- CUDA.newListArray [ 1.0, 2.0, 3.0
                            , 2.0, 3.0, 4.0
                            , 4.0, 5.0, 6.0
                            ]
  ptrB <- CUDA.newListArray [ 1.0, 2.0, 3.0, 2.0
                            , 2.0, -2.0, 4.0, 0.0
                            , 1.0, 5.0, 6.0, -1.0
                            ]
  ptrC <- CUDA.mallocArray (3 * 4)
  let m = 3
      n = 4
      k = 3
  with 1.0 $ \ptrAlpha ->
    with 0.0 $ \ptrBeta ->
      -- A: m * k
      -- B: k * n
      CuBLAS.dgemm handle CuBLAS.N CuBLAS.N n m k ptrAlpha ptrB n ptrA k ptrBeta ptrC n
  c <- CUDA.peekListArray (3 * 4) ptrC
  CUDA.free ptrA
  CUDA.free ptrB
  CUDA.free ptrC
  printMat 3 4 (VU.fromList c)
  CuBLAS.destroy handle
