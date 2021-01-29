import qualified Blas.Generic.Safe as BLAS
import qualified Blas.Primitive.Types as BLAS
import qualified Data.Vector.Storable as VS
import qualified Data.Vector.Storable.Mutable as VSM
import Control.Monad (forM_)

matMulVS :: Int -> Int -> Int -> VS.Vector Double -> VS.Vector Double -> IO (VS.Vector Double)
matMulVS m n k a b
  | VS.length a == m * k && VS.length b == k * n
  = VS.unsafeWith a $ \ptrA -> VS.unsafeWith b $ \ptrB -> do
      c <- VSM.new (m * n)
      VSM.unsafeWith c $ \ptrC ->
        BLAS.gemm BLAS.RowMajor BLAS.NoTrans BLAS.NoTrans m n k 1.0 ptrA k ptrB n 0.0 ptrC n
      VS.unsafeFreeze c
  | otherwise = error "size mismatch"

printMat :: Int -> Int -> VS.Vector Double -> IO ()
printMat m n v
  | VS.length v == m * n
  = forM_ [0..m-1] $ \i -> do
      print $ VS.slice (n * i) n v
  | otherwise = error "size mismatch"

main :: IO ()
main = do
  let a :: VS.Vector Double
      a = VS.fromList [ 1.0, 2.0, 3.0
                      , 2.0, 3.0, 4.0
                      , 4.0, 5.0, 6.0
                      ]

      b :: VS.Vector Double
      b = VS.fromList [ 1.0, 2.0, 3.0, 2.0
                      , 2.0, -2.0, 4.0, 0.0
                      , 1.0, 5.0, 6.0, -1.0
                      ]
  c <- matMulVS 3 4 3 a b
  printMat 3 4 c
