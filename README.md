# Quick-start to CUDA Runtime Code Generation 
## Environment
* CUDA toolkits installed.
* CUDA added to PATH.
## Step By Step
[NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html) is used for JIT code generation.
[jitify.hpp](./jitify.hpp) is a header-only template [library](https://github.com/NVIDIA/jitify) encapsulating NVRTC.
[main.cu](./main.cu) is an example of UDF(user defined function) on GPU. 
Users can provide a fragment of C++ code performing transformations on each **row** in [UDF](./UDF) file.
The program will read the UDF and dynamicly executed it on GPU in parallel.
### Build
```shell
mkdir build
cd build
cmake ..
cmake --build .
```
## Run
```shell
cd ..
./build/learn_cuda_jit
```
You can define your user-defined function in [UDF](./UDF) and run again without re-compilation.
## Insight
* Runtime code generation costs 200~400ms per kernel.