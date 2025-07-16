# Note
So these are from an old version of TinyGrad.  The new onnx runner has some odd behavoir that was killing performance.

With the new tinygrad native onnx stuff we get these `RUST <- PYTHON` sections that render as unsigned char instead of the floats they should be.
```
❯ DEBUG=3 uv run ml2code --model ./tmp/2024-11-23_20-42-27.onnx --test
warning: `VIRTUAL_ENV=tinygrad/.venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
opened device PYTHON from pid:769351
opened device NPY from pid:769351
JIT args=(<Tensor <UOp RUST (1, 32) float (<Ops.COPY: 6>, None)> on RUST with grad None>,) kwargs={}
opened device RUST from pid:769351
*** RUST       1 copy      128,    RUST <- NPY             arg  2 mem  0.00 GB tm    103.96us/     0.10ms (     0.00 GFLOPS    0.0|0.0     GB/s)
*** RUST       2 copy     8192,    RUST <- PYTHON          arg  2 mem  0.00 GB tm     57.10us/     0.16ms (     0.00 GFLOPS    0.1|0.1     GB/s)
*** RUST       3 copy      256,    RUST <- PYTHON          arg  2 mem  0.00 GB tm     29.98us/     0.19ms (     0.00 GFLOPS    0.0|0.0     GB/s)
*** RUST       4 copy     8192,    RUST <- PYTHON          arg  2 mem  0.00 GB tm     10.45us/     0.20ms (     0.00 GFLOPS    0.8|0.8     GB/s)
*** RUST       5 copy      128,    RUST <- PYTHON          arg  2 mem  0.00 GB tm      9.46us/     0.21ms (     0.00 GFLOPS    0.0|0.0     GB/s)
*** RUST       6 copy      512,    RUST <- PYTHON          arg  2 mem  0.00 GB tm     28.74us/     0.24ms (     0.00 GFLOPS    0.0|0.0     GB/s)
*** RUST       7 copy       16,    RUST <- PYTHON          arg  2 mem  0.00 GB tm     24.79us/     0.26ms (     0.00 GFLOPS    0.0|0.0     GB/s)
*** RUST       8 r_64_32                                   arg  4 mem  0.00 GB tm     21.70us/     0.29ms (     0.78 GFLOPS    0.4|0.4     GB/s) ['relu', '__add__', '__matmul__', '__rmul__', 'bitcast']
*** RUST       9 r_32_16_4                                 arg  4 mem  0.00 GB tm      7.40us/     0.29ms (     2.25 GFLOPS    1.2|2.2     GB/s) ['relu', '__add__', '__matmul__', '__rmul__', 'bitcast']
*** RUST      10 r_4_32                                    arg  4 mem  0.00 GB tm      6.43us/     0.30ms (     0.17 GFLOPS    0.1|0.1     GB/s) ['sigmoid', '__add__', '__matmul__', '__rmul__', 'bitcast']
JIT args=(<Tensor <UOp RUST (1, 32) float ShapeTracker(views=(View(shape=(1, 32), strides=(0, 1), offset=0, mask=None, contiguous=True),))> on RUST with grad None>,) kwargs={}
*** RUST      11 r_64_32                                   arg  4 mem  0.00 GB tm     11.13us/     0.31ms (     1.52 GFLOPS    0.8|0.8     GB/s) ['transpose', 'relu', '__add__', '__matmul__', '__rmul__']
*** RUST      12 r_32_16_4                                 arg  4 mem  0.00 GB tm      3.52us/     0.31ms (     4.74 GFLOPS    2.5|4.7     GB/s) ['relu', '__add__', '__matmul__', '__rmul__']
*** RUST      13 r_4_32                                    arg  4 mem  0.00 GB tm      3.52us/     0.32ms (     0.30 GFLOPS    0.2|0.2     GB/s) ['sigmoid', '__add__', '__matmul__', '__rmul__']
JIT captured 3 kernels with 1 inputs
----generate----------------
bufs={'buf_0': (256, dtypes.float, 21857004833008), 'input0': (128, dtypes.float, 21857006687328), 'buf_1': (8192, dtypes.uchar, 21857008693280), 'buf_2': (256, dtypes.uchar, 21857006335184), 'buf_3': (128, dtypes.float, 21857004830800), 'buf_4': (8192, dtypes.uchar, 21857006334800), 'buf_5': (128, dtypes.uchar, 21857006493024), 'output0': (16, dtypes.float, 21857004834448), 'buf_6': (512, dtypes.uchar, 21857006501232), 'buf_7': (16, dtypes.uchar, 21857006499888)}
bufs_to_save={'buf_1': <buf real:True device:RUST size:8192 dtype:dtypes.uchar>, 'buf_2': <buf real:True device:RUST size:256 dtype:dtypes.uchar>, 'buf_4': <buf real:True device:RUST size:8192 dtype:dtypes.uchar>, 'buf_5': <buf real:True device:RUST size:128 dtype:dtypes.uchar>, 'buf_6': <buf real:True device:RUST size:512 dtype:dtypes.uchar>, 'buf_7': <buf real:True device:RUST size:16 dtype:dtypes.uchar>}
line=      buf_0: [0.0; 64], dtypes.float
line=      buf_1: BUF_1_DATA, dtypes.uchar
line=      buf_2: BUF_2_DATA, dtypes.uchar
line=      buf_3: [0.0; 32], dtypes.float
line=      buf_4: BUF_4_DATA, dtypes.uchar
line=      buf_5: BUF_5_DATA, dtypes.uchar
line=      buf_6: BUF_6_DATA, dtypes.uchar
line=      buf_7: BUF_7_DATA, dtypes.uchar
----build----------------
   Compiling model v0.1.0 (/home/jared/repos/bitflux-ai/ml2code/export/model)
   Compiling model_test v0.1.0 (/home/jared/repos/bitflux-ai/ml2code/export/model_test)
    Finished `release` profile [optimized] target(s) in 0.39s
----test----------------
JIT args=(<Tensor <UOp RUST (1, 32) float (<Ops.COPY: 6>, None)> on RUST with grad None>,) kwargs={}
*** RUST      14 copy      128,    RUST <- NPY             arg  2 mem  0.00 GB tm     36.48us/     0.35ms (     0.00 GFLOPS    0.0|0.0     GB/s)
*** RUST      15 r_64_32                                   arg  4 mem  0.00 GB tm      8.44us/     0.36ms (     2.00 GFLOPS    1.0|1.0     GB/s) ['transpose', 'relu', '__add__', '__matmul__', '__rmul__']
*** RUST      16 r_32_16_4                                 arg  4 mem  0.00 GB tm      3.76us/     0.37ms (     4.44 GFLOPS    2.3|4.4     GB/s) ['relu', '__add__', '__matmul__', '__rmul__']
*** RUST      17 r_4_32                                    arg  4 mem  0.00 GB tm      4.14us/     0.37ms (     0.26 GFLOPS    0.2|0.2     GB/s) ['sigmoid', '__add__', '__matmul__', '__rmul__']
opened device CPU from pid:769351
*** CPU       18 copy       16,     CPU <- RUST            arg  2 mem  0.00 GB tm     24.43us/     0.40ms (     0.00 GFLOPS    0.0|0.0     GB/s)
Passed: torch == onnx
Passed: tiny == onnx
Passed: compiled == onnx
Passed:  True
binary at: export/model_test/target/release/model_test
```


With the this old version those end up as being `RUST <- NPY`
```
❯ DEBUG=3 uv run ml2code --model ./tmp/2024-11-23_20-42-27.onnx --test
warning: `VIRTUAL_ENV=tinygrad/.venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
Using LLVM at 'libLLVM.so.19.1'
opened device CUDA from pid:913844
dtype=dtypes.float  f
opened device NPY from pid:913844
dtype=dtypes.float  f
dtype=dtypes.float  f
dtype=dtypes.float  f
dtype=dtypes.float  f
dtype=dtypes.float  f
JIT args=(<Tensor <UOp RUST (1, 32) float (<Ops.COPY: 6>, None)> on RUST with grad None>,) kwargs={}
opened device RUST from pid:913844
*** RUST       1 copy      128,    RUST <- NPY             arg  2 mem  0.00 GB tm     54.49us/     0.05ms (     0.00 GFLOPS    0.0|0.0     GB/s)
inputs:
        onnx::Gemm_0 - <Tensor <UOp RUST (1, 32) float ShapeTracker(views=(View(shape=(1, 32), strides=(0, 1), offset=0, mask=None, contiguous=True),))> on RUST with grad None>  farts
        fc1.weight - <Tensor <UOp RUST (64, 32) float ShapeTracker(views=(View(shape=(64, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),))> on RUST with grad None>  farts
        fc1.bias - <Tensor <UOp RUST (64,) float (<Ops.COPY: 6>, None)> on RUST with grad None>  farts
inputs:
        /fc1/Gemm_output_0 - <Tensor <UOp RUST (1, 64) float (<Ops.ADD: 56>, None)> on RUST with grad None>  farts
inputs:
        /relu/Relu_output_0 - <Tensor <UOp RUST (1, 64) float (<Ops.WHERE: 73>, None)> on RUST with grad None>  farts
        fc2.weight - <Tensor <UOp RUST (32, 64) float ShapeTracker(views=(View(shape=(32, 64), strides=(64, 1), offset=0, mask=None, contiguous=True),))> on RUST with grad None>  farts
        fc2.bias - <Tensor <UOp RUST (32,) float (<Ops.COPY: 6>, None)> on RUST with grad None>  farts
inputs:
        /fc2/Gemm_output_0 - <Tensor <UOp RUST (1, 32) float (<Ops.ADD: 56>, None)> on RUST with grad None>  farts
inputs:
        /relu_1/Relu_output_0 - <Tensor <UOp RUST (1, 32) float (<Ops.WHERE: 73>, None)> on RUST with grad None>  farts
        fc3.weight - <Tensor <UOp RUST (4, 32) float ShapeTracker(views=(View(shape=(4, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),))> on RUST with grad None>  farts
        fc3.bias - <Tensor <UOp RUST (4,) float (<Ops.COPY: 6>, None)> on RUST with grad None>  farts
inputs:
        /fc3/Gemm_output_0 - <Tensor <UOp RUST (1, 4) float (<Ops.ADD: 56>, None)> on RUST with grad None>  farts
*** RUST       2 copy     8192,    RUST <- NPY             arg  2 mem  0.00 GB tm     48.82us/     0.10ms (     0.00 GFLOPS    0.2|0.2     GB/s)
*** RUST       3 copy      256,    RUST <- NPY             arg  2 mem  0.00 GB tm     23.10us/     0.13ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__rmul__']
*** RUST       4 copy     8192,    RUST <- NPY             arg  2 mem  0.00 GB tm     11.72us/     0.14ms (     0.00 GFLOPS    0.7|0.7     GB/s)
*** RUST       5 copy      128,    RUST <- NPY             arg  2 mem  0.00 GB tm     10.59us/     0.15ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__rmul__']
*** RUST       6 copy      512,    RUST <- NPY             arg  2 mem  0.00 GB tm     29.74us/     0.18ms (     0.00 GFLOPS    0.0|0.0     GB/s)
*** RUST       7 copy       16,    RUST <- NPY             arg  2 mem  0.00 GB tm     19.63us/     0.20ms (     0.00 GFLOPS    0.0|0.0     GB/s) ['__rmul__']
*** RUST       8 r_64_32                                   arg  4 mem  0.00 GB tm      7.41us/     0.21ms (     0.57 GFLOPS    1.2|1.2     GB/s) ['relu', '__add__', '__matmul__', '__rmul__']
*** RUST       9 r_32_16_4                                 arg  4 mem  0.00 GB tm      6.96us/     0.21ms (     0.60 GFLOPS    1.3|2.4     GB/s) ['relu', '__add__', '__matmul__', '__rmul__']
*** RUST      10 r_4_32                                    arg  4 mem  0.00 GB tm      6.50us/     0.22ms (     0.04 GFLOPS    0.1|0.1     GB/s) ['sigmoid', '__add__', '__matmul__', '__rmul__']
JIT args=(<Tensor <UOp RUST (1, 32) float ShapeTracker(views=(View(shape=(1, 32), strides=(0, 1), offset=0, mask=None, contiguous=True),))> on RUST with grad None>,) kwargs={}
inputs:
        onnx::Gemm_0 - <Tensor <UOp RUST (1, 32) float ShapeTracker(views=(View(shape=(1, 32), strides=(0, 1), offset=0, mask=None, contiguous=True),))> on RUST with grad None>  farts
        fc1.weight - <Tensor <UOp RUST (64, 32) float ShapeTracker(views=(View(shape=(64, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),))> on RUST with grad None>  farts
        fc1.bias - <Tensor <UOp RUST (64,) float (<Ops.BUFFER: 7>, <buf real:True device:RUST size:64 dtype:dtypes.float>)> on RUST with grad None>  farts
inputs:
        /fc1/Gemm_output_0 - <Tensor <UOp RUST (1, 64) float (<Ops.ADD: 56>, None)> on RUST with grad None>  farts
inputs:
        /relu/Relu_output_0 - <Tensor <UOp RUST (1, 64) float (<Ops.WHERE: 73>, None)> on RUST with grad None>  farts
        fc2.weight - <Tensor <UOp RUST (32, 64) float ShapeTracker(views=(View(shape=(32, 64), strides=(64, 1), offset=0, mask=None, contiguous=True),))> on RUST with grad None>  farts
        fc2.bias - <Tensor <UOp RUST (32,) float (<Ops.BUFFER: 7>, <buf real:True device:RUST size:32 dtype:dtypes.float>)> on RUST with grad None>  farts
inputs:
        /fc2/Gemm_output_0 - <Tensor <UOp RUST (1, 32) float (<Ops.ADD: 56>, None)> on RUST with grad None>  farts
inputs:
        /relu_1/Relu_output_0 - <Tensor <UOp RUST (1, 32) float (<Ops.WHERE: 73>, None)> on RUST with grad None>  farts
        fc3.weight - <Tensor <UOp RUST (4, 32) float ShapeTracker(views=(View(shape=(4, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),))> on RUST with grad None>  farts
        fc3.bias - <Tensor <UOp RUST (4,) float (<Ops.BUFFER: 7>, <buf real:True device:RUST size:4 dtype:dtypes.float>)> on RUST with grad None>  farts
inputs:
        /fc3/Gemm_output_0 - <Tensor <UOp RUST (1, 4) float (<Ops.ADD: 56>, None)> on RUST with grad None>  farts
*** RUST      11 r_64_32                                   arg  4 mem  0.00 GB tm      7.34us/     0.23ms (     0.58 GFLOPS    1.2|1.2     GB/s) ['transpose', 'relu', '__add__', '__matmul__', '__rmul__']
*** RUST      12 r_32_16_4                                 arg  4 mem  0.00 GB tm      3.39us/     0.23ms (     1.24 GFLOPS    2.6|4.9     GB/s) ['relu', '__add__', '__matmul__', '__rmul__']
*** RUST      13 r_4_32                                    arg  4 mem  0.00 GB tm      2.28us/     0.23ms (     0.12 GFLOPS    0.3|0.3     GB/s) ['sigmoid', '__add__', '__matmul__', '__rmul__']
JIT captured 3 kernels with 1 inputs
JIT memory reduced from 0.00 MB -> 0.01 MB, 2 -> 1 bufs
----generate----------------
bufs={'buf_0': (256, dtypes.float, 19982560240576), 'input0': (128, dtypes.float, 19982546148224), 'buf_1': (8192, dtypes.float, 19982546143616), 'buf_2': (256, dtypes.float, 19982546154368), 'buf_3': (128, dtypes.float, 19982560234720), 'buf_4': (8192, dtypes.float, 19982546152976), 'buf_5': (128, dtypes.float, 19982546155088), 'output0': (16, dtypes.float, 19982546149376), 'buf_6': (512, dtypes.float, 19982546153072), 'buf_7': (16, dtypes.float, 19982546149424)}
bufs_to_save={'buf_1': <buf real:True device:RUST size:2048 dtype:dtypes.float>, 'buf_2': <buf real:True device:RUST size:64 dtype:dtypes.float>, 'buf_4': <buf real:True device:RUST size:2048 dtype:dtypes.float>, 'buf_5': <buf real:True device:RUST size:32 dtype:dtypes.float>, 'buf_6': <buf real:True device:RUST size:128 dtype:dtypes.float>, 'buf_7': <buf real:True device:RUST size:4 dtype:dtypes.float>}
line=      buf_0: [0.0; 64], dtypes.float
line=      buf_1: BUF_1_DATA, dtypes.float
line=      buf_2: BUF_2_DATA, dtypes.float
line=      buf_3: [0.0; 32], dtypes.float
line=      buf_4: BUF_4_DATA, dtypes.float
line=      buf_5: BUF_5_DATA, dtypes.float
line=      buf_6: BUF_6_DATA, dtypes.float
line=      buf_7: BUF_7_DATA, dtypes.float
----build----------------
   Compiling model v0.1.0 (/home/jared/repos/bitflux-ai/ml2code/export/model)
   Compiling model_test v0.1.0 (/home/jared/repos/bitflux-ai/ml2code/export/model_test)
    Finished `release` profile [optimized] target(s) in 0.42s
----test----------------
JIT args=(<Tensor <UOp RUST (1, 32) float (<Ops.COPY: 6>, None)> on RUST with grad None>,) kwargs={}
*** RUST      14 copy      128,    RUST <- NPY             arg  2 mem  0.00 GB tm     40.68us/     0.27ms (     0.00 GFLOPS    0.0|0.0     GB/s)
*** RUST      15 r_64_32                                   arg  4 mem  0.00 GB tm      5.72us/     0.28ms (     0.74 GFLOPS    1.5|1.5     GB/s) ['transpose', 'relu', '__add__', '__matmul__', '__rmul__']
*** RUST      16 r_32_16_4                                 arg  4 mem  0.00 GB tm      3.98us/     0.28ms (     1.05 GFLOPS    2.2|4.2     GB/s) ['relu', '__add__', '__matmul__', '__rmul__']
*** RUST      17 r_4_32                                    arg  4 mem  0.00 GB tm      3.98us/     0.29ms (     0.07 GFLOPS    0.2|0.2     GB/s) ['sigmoid', '__add__', '__matmul__', '__rmul__']
opened device CPU from pid:913844
*** CPU       18 copy       16,     CPU <- RUST            arg  2 mem  0.00 GB tm     18.49us/     0.30ms (     0.00 GFLOPS    0.0|0.0     GB/s)
Passed: torch == onnx
Passed: tiny == onnx
Passed: compiled == onnx
Passed:  True
binary at: export/model_test/target/release/model_test
```

We can try to deal with this later.