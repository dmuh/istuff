using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.BLAS;
using System.Diagnostics;
namespace MatMul
{
    public class MatrixMultClass
    {
        public const int BLOCK_SIZE = 32;

        public static void MatrixMultiplication(float[] d_C, float[] d_A, float[] d_B, int matrixWidth, GPGPU gpu)
        {
            MatrixMultiplicationGPU(d_C, d_A, d_B, matrixWidth, gpu, false);
        }
        
        public static void MatrixMultiplicationBLAS(float[] d_C, float[] d_A, float[] d_B, int matrixWidth, GPGPU gpu)
        {
            MatrixMultiplicationGPU(d_C, d_A, d_B, matrixWidth, gpu, true);
        }
        
        private static void MatrixMultiplicationGPU(float[] d_C, float[] d_A, float[] d_B, int matrixWidth, GPGPU gpu, bool useBlas)
        {

            if (useBlas)
            {
                GPGPUBLAS blas = GPGPUBLAS.Create(gpu);
                blas.GEMM(matrixWidth, matrixWidth, matrixWidth, 1.0f, d_B, d_A, 0.0f, d_C);
            }
            else
            {
                dim3 block = new dim3(32, 32, 1);
                dim3 grid = new dim3(matrixWidth / 32, matrixWidth / 32);
                gpu.Launch(grid, block, ((Action<GThread, float[], float[], float[], int>)MatMultKernel), d_C, d_A, d_B, matrixWidth);
            }
        }

        [Cudafy]
        private static void MatMultKernel(GThread thread, float[] C, float[] A, float[] B, int width)
        {
                // Block index
            int bx = thread.blockIdx.x;
            int by = thread.blockIdx.y;

            // Thread index
            int tx = thread.threadIdx.x;
            int ty = thread.threadIdx.y;

            // Index of the first sub-matrix of A processed by the block
            int aBegin = width * BLOCK_SIZE * by;

            // Index of the last sub-matrix of A processed by the block
            int aEnd   = aBegin + width - 1;

            // Step size used to iterate through the sub-matrices of A
            int aStep  = BLOCK_SIZE;

            // Index of the first sub-matrix of B processed by the block
            int bBegin = BLOCK_SIZE * bx;

            // Step size used to iterate through the sub-matrices of B
            int bStep  = BLOCK_SIZE * width;

            // Csub is used to store the element of the block sub-matrix
            // that is computed by the thread
            float Csub = 0;

            // Loop over all the sub-matrices of A and B
            // required to compute the block sub-matrix
            for (int a = aBegin, b = bBegin;
                 a <= aEnd;
                 a += aStep, b += bStep)
            {

                // Declaration of the shared memory array As used to
                // store the sub-matrix of A
                float[,] As = thread.AllocateShared<float>("As", BLOCK_SIZE, BLOCK_SIZE);

                // Declaration of the shared memory array Bs used to
                // store the sub-matrix of B

                 float[,] Bs = thread.AllocateShared<float>("Bs", BLOCK_SIZE, BLOCK_SIZE);
                // Load the matrices from device memory
                // to shared memory; each thread loads
                // one element of each matrix
                As[ty,tx] = A[a + width * ty + tx];
                Bs[ty,tx] = B[b + width * ty + tx];

                // Synchronize to make sure the matrices are loaded
                thread.SyncThreads();

                // Multiply the two matrices together;
                // each thread computes one element
                // of the block sub-matrix
                for (int k = 0; k < BLOCK_SIZE; ++k)
                {
                    Csub += As[ty,k] * Bs[k,tx];
                }

                // Synchronize to make sure that the preceding
                // computation is done before loading two new
                // sub-matrices of A and B in the next iteration
                thread.SyncThreads();
            }

            // Write the block sub-matrix to device memory;
            // each thread writes one element
            int c = width * BLOCK_SIZE * by + BLOCK_SIZE * bx;
            C[c + width * ty + tx] = Csub;
        }
    }
}
