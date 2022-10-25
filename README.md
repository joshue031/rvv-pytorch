# rvv-pytorch

## Building

```
mkdir build
cd build
cmake ..
make
``

The built library can be loaded into Pytorch with `torch.ops.load_library("librvv_pytorch.so")` (see the example `rvv_fcnet.py` ).
