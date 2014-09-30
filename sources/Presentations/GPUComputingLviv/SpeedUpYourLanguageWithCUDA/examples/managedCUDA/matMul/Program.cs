/*
 * This code is based on code from the NVIDIA CUDA SDK. (Ported from C++ to C# using managedCUDA)
 * This software contains source code provided by NVIDIA Corporation.
 *
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.IO;
using System.Diagnostics;
using System.Threading;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.CudaBlas;

namespace matMul
{
	class Program
	{
		static CudaContext ctx;
		static Random rand = new Random();
		//static bool noprompt;
		// Variables
		static float[] h_A;
		static float[] h_B;
		static float[] h_C;
		static CudaDeviceVariable<float> d_A;
		static CudaDeviceVariable<float> d_B;
		static CudaDeviceVariable<float> d_C;

		static void Main(string[] args)
		{
			Console.WriteLine("Matrix Multiplication with CUDA Kernel");
            int matrixWidth = 640;
			int N = matrixWidth * matrixWidth;
            
            //Init Cuda context
			ctx = new CudaContext(CudaContext.GetMaxGflopsDeviceId());

			//Load Kernel image from resources
			string resName;
			if (IntPtr.Size == 8)
				resName = "matMul_x64.ptx";
			else
				resName = "matMul.ptx";

			string resNamespace = "matMul";
			string resource = resNamespace + "." + resName;
			Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resource);
			if (stream == null) throw new ArgumentException("Kernel not found in resources.");

            CudaKernel matMulKernel = ctx.LoadKernelPTX(stream, "matrixMulCUDA32");

			// Allocate input vectors h_A and h_B in host memory
			h_A = new float[N];
			h_B = new float[N];
			

			// Initialize input vectors
            ConstInit(h_A, N, 1.0f);
            ConstInit(h_B, N, 0.01f);
			
			// Allocate vectors in device memory and copy vectors from host memory to device memory 
			// Notice the new syntax with implicit conversion operators: Allocation of device memory and data copy is one operation.
			d_A = h_A;
			d_B = h_B;
			d_C = new CudaDeviceVariable<float>(N);

			// Invoke kernel
			matMulKernel.BlockDimensions = new dim3(32, 32);
			matMulKernel.GridDimensions = new dim3(matrixWidth/32, matrixWidth/32);
            // Create new stopwatch
            Stopwatch stopwatch = new Stopwatch();
            CudaEvent start = new CudaEvent();
            CudaEvent end = new CudaEvent();
            matMulKernel.Run(d_C.DevicePointer, d_A.DevicePointer, d_B.DevicePointer, matrixWidth, matrixWidth);
            ctx.Synchronize();
            stopwatch.Start();
            start.Record();
            float cudaTime_ms = matMulKernel.Run(d_C.DevicePointer, d_A.DevicePointer, d_B.DevicePointer, matrixWidth, matrixWidth);
            end.Record();
            end.Synchronize();
            stopwatch.Stop();

			// Copy result from device memory to host memory
			// h_C contains the result in host memory
			h_C = d_C;

			// Verify result
			int i;
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
                Console.WriteLine("Pass with {0} ms", stopwatch.ElapsedMilliseconds);
                Console.WriteLine("Cuda Events Time: {0} ms", CudaEvent.ElapsedTime(start, end));
            }

            h_C = h_A;
            //cublass
            CudaBlasHandle cuHandle= new CudaBlasHandle();
            CudaBlasNativeMethods.cublasCreate_v2(ref cuHandle);
            float alpha = 1;
            float beta = 0;
            start.Record();
            CublasStatus status = CudaBlasNativeMethods.cublasSgemm_v2(cuHandle, Operation.NonTranspose, Operation.NonTranspose, matrixWidth, matrixWidth, matrixWidth, ref alpha, d_B.DevicePointer, matrixWidth, d_A.DevicePointer, matrixWidth, ref beta, d_C.DevicePointer, matrixWidth);
            end.Record();
            end.Synchronize();
            if (status != CublasStatus.Success)
            {
                Console.WriteLine("CuBlas Failed");
            }
            h_C = d_C;
            // Verify result
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
                Console.WriteLine("CuBlas Pass");
                Console.WriteLine("Cuda Events Time: {0} ms", CudaEvent.ElapsedTime(start, end));
            }
            CleanupResources();
		}

		
		static void CleanupResources()
		{
			// Free device memory
			if (d_A != null)
				d_A.Dispose();

			if (d_B != null)
				d_B.Dispose();

			if (d_C != null)
				d_C.Dispose();

			if (ctx != null)
				ctx.Dispose();

			// Free host memory
			// We have a GC for that :-)
		}

		// Allocates an array with random float entries.
		static void RandomInit(float[] data, int n)
		{
			for (int i = 0; i < n; ++i)
				data[i] = (float)rand.NextDouble();
		}
        
        static void ConstInit(float[] data, int n, float constant)
        {
            for (int i = 0; i < n; ++i)
                data[i] = constant;
        }
	}
}
