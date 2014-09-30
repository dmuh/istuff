import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
from jinja2 import Template

start, end = cuda.Event(), cuda.Event()
matrixWidth = 640
matrixSize = matrixWidth*matrixWidth
matrixA = np.full(matrixSize, 1.0, dtype=np.float32)
matrixB = np.full(matrixSize, 0.01, dtype=np.float32)

matrixC = np.full(matrixSize, 0.0, dtype= np.float32)
blockWidth = 32
blockDim = (blockWidth, blockWidth, 1)
gridDim = (matrixWidth/blockDim[0],matrixWidth/blockDim[1],1)
kernelStr = """
template <int BLOCK_SIZE, typename realT>
__device__ void
matrixMulCUDA(realT *C, realT *A, realT *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

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
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

extern "C" 
{
{% for blockSize in blocks %}
	__global__ void callMatrixMulCuda{{blockSize}}({{realT}} *C, {{realT}} *A, {{realT}} *B, int wA, int wB)
	{
		matrixMulCUDA<{{blockSize}}>(C, A, B, wA, wB);
	}
{% endfor %}
}
"""

template = Template(kernelStr)
renderedKernelSource = template.render(blocks=(16, 32), realT="float")
print (renderedKernelSource)

func_mod = SourceModule(renderedKernelSource, no_extern_c=1)

func = func_mod.get_function('callMatrixMulCuda'+str(blockWidth))
a_gpu = gpuarray.to_gpu(matrixA)
b_gpu = gpuarray.to_gpu(matrixB)
c_gpu = gpuarray.to_gpu(matrixC)

start.record()
func(c_gpu, a_gpu, b_gpu, np.uint32(matrixWidth),np.uint32(matrixWidth),block=blockDim,grid=gridDim)
end.record() 
end.synchronize()
mseconds = start.time_till(end)

print ("SourceModule time :")
print ("%f ms" % (mseconds))

c_dev_result=c_gpu.get()

print c_dev_result[0] 

error = False
for i in xrange(matrixSize):
	abs_err = abs(c_dev_result[i] - (matrixWidth * 0.01));
	dot_length = matrixWidth;
	abs_val = abs(c_dev_result[i]);
	rel_err = abs_err/abs_val/dot_length ;

	if (rel_err > (0.1**5)):
		print ("Fail!!! in", i, "index")
		error = True
		break

if not error:
	print ("Result = PASS")

