1. Make sure that the python model is working with torchScript. Calling jittable everywhere is not enough. Need to remove Optional Tensors and optional arguments.
2. Install Cuda 11.6 and Cudnn v8.4.1
3. go to cuda/bin and change g++ symlink to g++-11 and gcc symlink to gcc-11
4. get libtorch with cuda 11.6 support from pytorch website
5. clone pytorch-scatter and pytorch-sparse repos with --recursive
6. build and install both torch-sparse and torch-scatter, both with WITH\_CUDA=on and link to libtorch
7. Create CMake file similar to the one here.
