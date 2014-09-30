This is a collection of CUDA samples from the NVIDIA CUDA SDK ported to C# using the ManagedCuda library.

The base library, ManagedCuda.dll, is compiled as "Any CPU", i.e. the library can eather run as a 32-bit 
or a 64-bit application. Whereas the library is architecture independent, Cuda kernels are architecture 
specific. Have a look at the simple "vectorAdd" sample how this can be handled at runtime.

As NVidia provides different names for 32/64 bit Cuda libraries such as cufft or NPP, the wrappers must 
also be architecture specific (constant library name must be known at compile time). .Net applications 
using a wrapped Cuda library should be compiled to either 32-bit or 64-bit and be linked to the 
correspondent library to ensure that the application always uses the right library.

All Fluids* samples (using CudaFFT) and NPP samples are x86 only. If you want them 64 bit, compile them 
with a reference to a 64 bit library assembly.