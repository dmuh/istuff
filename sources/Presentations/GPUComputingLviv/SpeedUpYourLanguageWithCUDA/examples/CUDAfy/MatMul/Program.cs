using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Cudafy;
using Cudafy.Host;
using Cudafy.Maths.BLAS;
using Cudafy.Translator;
using System.Diagnostics;
using System.Threading;

namespace MatMul
{
    class Program
    {

        static void Main(string[] args)
        {
            // Check we have at least one CUDA GPU.
            int cnt = CudafyHost.GetDeviceCount(eGPUType.Cuda);
            if (cnt < 1)
            {
                Console.WriteLine("No CUDA Capable devices found, exiting...");
                return;
            }

            Console.WriteLine("Matrix Multiplication with CUDAfy Kernel");
            int matrixWidth = 640;
            int N = matrixWidth * matrixWidth;
            

            
            float[] h_A = new float[N];
            float[] h_B = new float[N];
            float[] h_C = new float[N];


            int i;

            // Get the GPU object and check that it supports double precision.
            GPGPU gpu = CudafyHost.GetDevice();
            if (gpu.GetArchitecture() < eArchitecture.sm_13)
            {
                Console.WriteLine("Double precision is not supported.");
                return;
            }

            // The magic happens here. If the program has already run there will be GPU module already saved to disk. Try loading this.
            CudafyModule module = CudafyModule.TryDeserialize();
            if (module == null || !module.TryVerifyChecksums())
            {
                // We need to CUDAfy and translate the C# code to CUDA C and complile using the NVIDIA compiler nvcc
                module = CudafyTranslator.Cudafy(ePlatform.Auto, gpu.GetArchitecture(), typeof(MatrixMultClass));
                // Cache for next time
                module.TrySerialize();
            }
            // Load the GPU module
            gpu.LoadModule(module);

            Console.WriteLine("Generating input data...");
            //Generate options set
            ConstInit(h_A, N, 1.0f);
            ConstInit(h_B, N, 0.01f);
            float[] d_A = gpu.CopyToDevice(h_A);
            float[] d_B = gpu.CopyToDevice(h_B);
            float[] d_C = gpu.Allocate<float>(matrixWidth * matrixWidth);

            MatrixMultClass.MatrixMultiplication(d_C, d_A, d_B, matrixWidth, gpu);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            MatrixMultClass.MatrixMultiplication(d_C, d_A, d_B, matrixWidth, gpu);
            sw.Stop();
            gpu.CopyFromDevice(d_C, 0, h_C, 0, h_C.Length);
            Console.WriteLine("Comparing the results...");
            // Verify result

            Console.WriteLine("Pass with {0} ms", sw.ElapsedMilliseconds);
            bool error = false;
            for (i = 0; i < N; ++i)
            {
                if (Math.Abs(h_C[i] - 0.01f * matrixWidth) > 1e-4)
                {
                    Console.WriteLine("{0} value: {1} diff: {2}", i, h_C[i], Math.Abs(h_C[i] - 0.01f * matrixWidth));
                    error = true;
                    break;
                }
            }
            if (!error)
            {
                Console.WriteLine("Correct");
            }
            h_C = h_A;
            MatrixMultClass.MatrixMultiplicationBLAS(d_C, d_A, d_B, matrixWidth, gpu);
            sw.Restart();
            MatrixMultClass.MatrixMultiplicationBLAS(d_C, d_A, d_B, matrixWidth, gpu);
            sw.Stop();
            Console.WriteLine("BLAS Pass with {0} ms", sw.ElapsedMilliseconds);
            gpu.CopyFromDevice(d_C, 0, h_C, 0, h_C.Length);
            error = false;
            for (i = 0; i < N; ++i)
            {
                if (Math.Abs(h_C[i] - 0.01f * matrixWidth) > 1e-4)
                {
                    Console.WriteLine("{0} value: {1} diff: {2}", i, h_C[i], Math.Abs(h_C[i] - 0.01f * matrixWidth));
                    error = true;
                    break;
                }
            }
            if (!error)
            {
                Console.WriteLine("Correct");
            }
            Console.WriteLine("Shutting down...");
            gpu.FreeAll();

        }
        static void ConstInit(float[] data, int n, float constant)
        {
            for (int i = 0; i < n; ++i)
                data[i] = constant;
        }

    }
}
