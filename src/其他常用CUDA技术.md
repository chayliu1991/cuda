# 其他常用CUDA技术

## CUDA API

CUDA中可调用的API分为runtime API和driver API。

**CUDA driver API** 是一种基于句柄的底层接口，可以加载二进制或汇编形式的kernel模块，指定参数并启动计算。CUDA driver API编程复杂，但有时能通过直接操作硬件的执行实现一些更为复杂的功能，或者获得更高的性能。由于它使用的device代码是二进制或者汇编代码，因此可以在各种语言中调用。CUDA driver API被存放在nvCUDA包里，所有函数前缀为cu。

**CUDA runtime API ** 则是在CUDA driver API的基础上进行了封装，隐藏了一些实现的细节，编程更加方便，代码更加简洁。CUDA runtime API被打包存放在CUDAart包里，其中的函数都有cuda前缀，调用时需要包含头文件 `cuda_runtime.h`。CUDA运行时没有专门的初始化函数，它将在第一次调用runtime函数时自动完成初始化。对使用runtime函数的CUDA程序测试时要避免将这段初始化时间计入。

## Stream

stream可以理解为流水线，是CUDA中非常重要的一个技术。每一个stream就是一个独立的流水线，通过stream可以并发多个kernel、多个API等。每个stream里的任务串行地先后被执行，而不同stream间的任务则可以并行执行（甚至是乱序执行），这可以根据硬件、程序设计等情况确定并行执行的具体部分。

下面的程序给了使用一个stream的示例。其中，stream通过创建cudaStream_t对象定义，并在启动kernel和Memcpy时将该对象作为参数传入。参数相同的属于同一个流，参数不同的属于不同的流。

```
int main(int argc, char** argv)
{
  // create 2 streams
  cudaStream_t stream[2];
  // initialize cuda stream
  for (int i = 0; i < 2; i++) {
    cudaStreamCreate(&stream[i]);
  }
 
  // alloc pinned memory and device memory;
  float *hostPtr, *d_in, *d_out;
  size_t size = 512 * sizeof(float);
  cudaMallocHost((void **)hostPtr, 2 * size);
  cudaMalloc((void **)d_in,  2 * size);
  cudaMalloc((void **)d_out, 2 * size);
 
  // copy data from host to device in 2 streams
  for (int i = 0; i < 2; i++) {
    cudaMemcpyAsync(d_in + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevce, stream[i]);
  }
 
  // run testkernel in 2 streams
  for (int i = 0; i < 2; i++) {
    testkernel<<<1, 512, 0, stream[i]>>>(d_in + i * size, d_out + i * size, size);
  }
 
  // copy data from device to host in 2 streams
  for (int i = 0; i < 2; i++) {
    cudaMemcpyAsync(hostPtr + i * size, d_out + i * ptr, size, cudaMemcpyDevceToHost, stream[i]);
  }
 
  cudaThreadSynchronize();
 
  // release cuda stream
  for (int i = 0; i < 2; i++) {
    cudaStreamDestroy(stream[i]);
  }
  cudaFree(hostPtr);
 
  return 0;
}
```

上面的代码将每个stream定义为host到device的内存拷贝、kernel启动以及device到host的内存拷贝三个操作组成的序列。每个流拷贝hostPtr的一部分到显存中的d_in数组，然后调用testkernel对d_in中的数据处理后存至d_out，并将计算结果从d_out拷贝回hostPtr。当使用两个stream处理hostPtr时，允许一个stream的内存拷贝与另一个流的kernel执行同时进行，这样就隐藏了host↔device通信时间。最后，cudaThreadSynchronize()保证了下一步操作前两个流都已经运行完毕，而cudaStreamSynchronize()可以用于主机与某一个特定流的同步。

在cudaMemcpyAsync中，stream[i]作为最后一个参数传入，在testkernel的“<<< >>>”中，stream[i]是第四个参数（第三个是使用共享内存的大小）。

## Event

CUDA中event用于在stream的执行中添加标记点，用于检查正在执行的流是否到达给定点。这样能起到三个作用：

- event可用于等待和测试时间插入点前的操作，作用和cudaStreamSynchronize类似。
- event可插入不同的流中，用于流之间的操作。不同流执行是并行的，特殊情况下，需要同步操作。同样，也可以在主机端操控设备端执行情况。
- 可以用于统计时间，在需要测量的函数前后插入event。调用cudaEventElapseTime()查看时间间隔。

下面代码是利用event计时的示例。其中event通过创建cudaEvent_t对象定义，释放通过cudaEventDestroy()进行。

```
int main(int argc, char** argv)
{
  // create two events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
 
  // record start event on the default stream
  cudaEventRecord(start);
 
  // execute kernel
  testkernel<<<1, 1024>>>();
 
  // record stop event on the default stream
  cudaEventRecord(stop);
 
  // wait until the stop event completes
  cudaEventSynchronize(stop);
 
  // calculate the elapsed time between two events
  float time;
  cudaEventElapsedTime(&time, start, stop);
 
  // clean up the two events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
```

## Dynamic parallelism

在较新版本的GPU（sm_35以上）和CUDA（CUDA 5.0以上）中，支持动态并行技术，这支持kernel由自身创建调用。

动态并行使递归等算法更容易实现和理解，由于启动的配置可以由kernel运行时的各个thread决定，这也减少了host和device之间传递数据和执行控制。通过动态并行性，可以直到kernel运行时才推迟确定在GPU上创建有多少块和网格，利用GPU硬件调度器和负载平衡动态地适应数据驱动的决策或工作负载。













