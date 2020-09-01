# CUDA通信机制

在CUDA编程中，涉及到各种同步函数，包括block内thread同步、block间thread同步、GPU与CPU间同步等。

## 同步函数

线程同步（也可以理解为在同步位置有一个barrier阻挡先到的线程，只有所有满足条件的线程都到这个barrier时，这些线程才会继续进行barrier之后的任务）是并行算法必须考虑的问题，在线程间有数据依赖或运算先后关系时，不做同步会得到错误的结果。比如当一些thread需要访问同一个内存地址时，可能会发生读后写、写后读、写后写错误。

CUDA中也有各种功能的同步函数适应各种需求。

### block 内的threads同步

__syncthreads()实现了block内所有thread的同步，保证block内所有thread都先执行到同一位置，之后再继续执行后面的代码。这样才能保证之前语句执行结果对block内所有thread可见。

需要注意的是，一般来说，不应在条件语句的分支中使用 `__syncthreads()`，这是因为不进入存在`__syncthreads()`的分支的thread，不会执行这个同步而继续执行后续代码，所以进入这个分支的thread就会因为`__syncthreads()`没有等到全部的thread而一直卡在同步位置。但是，如果确保整个block都走向相同分支，则此时分支内是可以使用`__syncthreads()`的。另外，由于block内的thread是按warp的方式执行的，warp内的线程无需调用`__syncthreads()`同步。

在最理想的情况下（所有线程都同时抵达barrier处），调用一次`__syncthreads()`需要至少四个时钟周期（SM内有8个SP同时执行8个warp，一个block内最多32个warp）。但一般调用`__syncthreads()`都需要更多的时钟周期，因此，要尽量避免或节约使用`__syncthreads()`。

在下面的kernel中，如果不加入`__syncthreads()`，线程不会检查block[j][i]处的数据是否已经被其他warp中的线程写入新值，而立即将block[j][i]处的数据写入global memory，这样就造成了读后写错误。所以，block内线程在向outdata写数据之前，必须调用一次`__syncthreads()`，才能保证所有线程都从inputdata中正确读入数据。

```
__global__ void transpose(float **inputdata, float **outputdata, int width, int height)
{
  __shared__ float block[32][32]; // alloc static shared memory, blockDim.x = 32, blockDim.y = 32
 
  // read matrix to shared memory
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
 
  if ((x >= width) || (y >= height)) {
    return;
  }
 
  int i = threadIdx.x, j = threadIdx.y;
 
  block[j][i] = inputdata[x][y];
 
  __syncthreads();
 
  outputdata[x][y] = block[i][j];
}
```

### global memoey 和 shared memory 操作后的threads同步

- `__threadfence_system()`：不同GPU内global memory与shared memory操作后所有thread同
- `__threadfence_block()`：block内global memory与shared memory操作后thread同步
- `__threadfence()`：grid内global memory与shared memory操作后thread同步

`__threadfence_system()`，`__threadfence_block()`，`__threadfence()` 三个memory fence函数是用来保证线程间数据通信的可靠性的。与__syncthreads()不同，memory fence函数并不要求所有线程都运行到同一位置，而只保证执行memory fence函数的线程产生的数据能够安全地被其他线程消费，从而多个线程间可以正确地操作共享数据，实现grid/block甚至多个GPU内的线程间通信。

这三个memory fence函数作用的范围有小到大分别是block内、grid内和多个GPU内，它们分别保证该thread在该语句前对global memory和shared memory的访问已经全部完成，执行的结果分别对block内、grid内和全部GPU内的thread可见。相对而言，`__threadfence()`和`__threadfence_block()`更为常用，只有在支持nvlink等GPU间直接通讯的场景下，__threadfence_system()才会发挥作用。

```
__device__ int count = 0;
  
__global__ static void sum(int* data_gpu, int* block_gpu, int *sum_gpu, int length)
{
  extern __shared__ int blocksum[];
  __shared__ int islast;
  int offset;
  
  const int tid  = threadIdx.x;
  const int bid  = blockIdx.x;
  const int tnum = blockDim.x;
  const int bnum = gridDim.x;
  blocksum[tid]  = 0;
 
  for (int i = bid * tnum + tid; i < length; i += bnum * tnum) {
    blocksum[tid] += data_gpu[i];
  }
  
  __syncthreads();
 
  offset = tnum / 2;
  while (offset > 0) {
    if(tid < offset) {
      blocksum[tid] += blocksum[tid + offset];
    }
    offset >>= 1;
    __syncthreads();
  }
  
  if (tid == 0) {
    block_gpu[bid] = blocksum[0];
    __threadfence();
  
    int value = atomicAdd(&count, 1);
    islast = (value == gridDim.x - 1);
  }
  
  __syncthreads();
  
  if (islast) {
    if (tid == 0) {
      int s = 0;
      for (int i = 0; i < bnum; i++) {
        s += block_gpu[i];
      }
      *sum_gpu = s;
    }
  }
}
```

上述CUDA代码实现了block之间对元素求和，threadfence()和原子操作atomicAdd()均进行了同步。通过分别去除`__threadfence()`和后面的atomicAdd()来验证结果的正确性，结果发现：单独的`__threadfence`不能给出正确结果；只用原子操作可以给出正确结果。这是因为`__threadfence()`不是保证所有线程都完成同一操作，而只保证正在进行fence的线程本身的操作能够对所有线程安全可见，memory fence不要求线程运行到同一指令，而barrier有要求。上述结论指出`__threadfence()`函数不同于同步函数`__syncthreads()`，如果单纯地让block 0去计算最终的结果，这时可能会存在还有其他block尚未执行，这时得到的结果必然是错误的。

虽然只用atomicAdd()可以给出正确结果，但是也不能保证在其他情况下也是正确的（GPU编程需要特别注意当前条件下正确的程序换个条件不一定正确，反映了GPU编程的复杂性）。这里正确的原因可能是因为访问的global memory只有一个空间，原子操作也是访问global memory中的变量，这两个访问时间属于一个量级导致。如果一开始的访问global memory不是一个空间，而是一个比较长的数组，则此时就可能会出错。为什么会出错，这和CUDA对global memory的读写有关：thread在读取global memory的时候会被阻塞，然后warp scheduler会接着调度其他warp；但是当线程在写入global memory时，虽说该写入操作尚未完成，但是线程会接着执行下面的指令，而不是等待写入完成。在这种情况下，如果访问不同block中的数据，不加`__threadfence()`确实会存在出错的可能。

现在重新考虑一种情况，在atomicAdd()之前的global memory方法写入的是一个长数组，然后我们去掉`__threadfence()`，只用原子操作来保证正确性。此时最后对原子变量进行操作的block完成之后开始对global memory进行读操作。由于block调度的不确定性，这时可能会存在其他block中的线程尚未完成global memory的写入，此时访问其他block要写入的global memory就会出错，所以`__threadfence()`是必须的。

通过运行代码和上面的分析我们可以看出`__threadfence()`对block之间的global memory或者shared memory访问的重要性，同时也可以看出原子操作是保证正确性必不可少的部分。它们两个组合就可以解决block之间的内存访问问题。这种方式有个好处：解决block之间的内存访问问题可以在一个kernel内完成，减少kernel的调用。

### CPU 与 GPU 之间的同步

- `__cudaDeviceSynchronize()`：CPU与GPU间同步
- `__cudaThreadSynchronize()`：CPU与GPU间同步，未来可能会被`__cudaDeviceSynchronize()`替代
- `__cudaStreamSynchronize()`：CPU与GPU间流同步
- `__cudaEventSynchronize()`：CPU与GPU间事件同步

这一类函数实现了CPU和GPU间的同步，在CPU线程中调用。其功能相当于在CPU端设置了一个barrier，CPU线程运行到该barrier时，只有之前所有的CUDA调用均已完成，CPU线程才会继续运行后面的代码。

通过这种方式，同样是保证了类似同步函数`__syncthreads()`可达到的正确性要求。下面是一个通过`__cudaThreadSynchronize()`函数同步CPU和GPU的例子：由于`__cudaThreadSynchronize()`函数的存在，只有当kernel1函数的两个block全部计算完毕后CPU才会执行 `int b = 4 * 256` 和kernel2。但是如果kernel1的执行时间稍长，在同步前，CPU有充裕的时间执行完 `int a = 2 * 512` 再等待GPU完成kernel1的计算。

```
int main(int argc, char** argv)
{
  // some initialization
 
  kernel1<<<2, 512>>>();
 
  int a = 2 * 512;
 
  __cudaThreadSynchronize(); // or __cudaDeviceSynchronize();
 
  int b = 4 * 256;
 
  kernel2<<<4, 256>>>();
 
  return 0;
}
```

CPU线程同步函数的存在，从另一个角度来看，则是提供了异步执行CPU端和GPU端代码的机会，比如在CPU等待GPU计算的时候，可以再执行一些其他代码（如上面代码中的 `int a = 2 * 512;`），这样能够更充分地利用计算资源。

## volatile关键字

```
// myArray是存储在全局或者共享存储器中的数组，元素是非零元素
__global__ void myKernel1(int* result)
{
    int tid = threadIdx.x;
    int ref1 = myArray[tid] * 1;
 
    myArray[tid + 1] = 2;
 
    int ref2 = myArray[tid] * 1;
    result[tid] = ref1 * ref2;
}
```

在这段kernel中，对myArray[tid]的首次引用将会去做一次对global或shared memory的访问。但第二次引用时，因为编译器并不知道程序对myArray[tid]的值进行了修改，它只知道程序都是在引用myArray数组的相同位置的值，所以它会直接使用上一次读出的结果。这也是编译器为了提高代码执行效率所做的一个优化。上述代码中ref1代表当前线程第一次引用，ref2代表当前线程第二次引用。于是，线程tid中的ref2的值不会是线程tid-1修改的myArray[tid]之后的值2。也就是，假设myArray中的值初始化为1，当tid=1时，ref1为1。然后因为线程0在ref1之后已经将myArray[1]的值修改为2，预期的ref2本应是2，即myArray[1]的值应该是2。但是由于编译器优化，得到的ref2的值其实是1。

这显然这不是我们想要的结果，而volatile关键字可以改变这个缺陷。使用volatile关键字可以把myArray声明为敏感变量，让编译器认为其它线程可能随时会修改该变量的值，从而每次对该变量的引用都会被编译成一次真实的内存读指令。

注意，即使将myArray使用volatile关键字进行声明，仍然不能保证第二次引用时的值为2，因为线程tid可能在myArray[tid]被线程tid-1改写为2之前就已经进行了读操作。此时需要进行一次同步才能保证正确性，可以参考以下代码实现：

```
// myArray共享存储器中的数组，元素是非零元素，并且已经通过volatile关键字声明
__global__ void myKernel2(int* result)
{
    __shared__ volatile float myArray[N]; // N是一个宏，表示数组的大小
 
    int tid = threadIdx.x;
    int ref1 = myArray[tid] * 1;
 
    myArray[tid + 1] = 2;
 
    __syncthreads();  // 同步操作，保证在第二次读myArray[tid]之前，此位置的值已经被写为了2
 
    int ref2 = myArray[tid] * 1;
    result[tid] = ref1 * ref2;
}
```

## atom操作

atom即大名鼎鼎的原子操作，少了它并行程序几乎无法正确实现。其核心就是当多个thread同时访问global或shared memory中的同一位置时，保证thread能够实现对共享可写操作的互斥操作，在一个操作完成之前，其他任何thread都无法访问此地址。

## vote操作

vote操作综合考虑一个warp内（注意不是block内）所有的thread，根据这些thread的不同数值表决出一个结果。vote操作可以通过以下两个函数调用：

- `int  __all(int predicate)`：将predicate与0进行比较，如果当前线程所在的wrap内所有线程对应的predicate不为0，则返回1。
- `int  __any(int predicate)`：将predicate与0进行比较，如果当前线程所在的wrap内有一个线程对应的predicate值不为0，则返回1。

在PTX（可以在GPU端执行的汇编指令）中还提供了另一个vote指令：`vote.uni`。即当一个warp中所有thread的判断表达式同时为1或者同时为0时，返回1，否则返回0。这相当于对所有结果做异或运算。

下面的代码给出了两个vote函数的应用。

```
__global__ void vote_all(int* a, int* b, int n) 
{ 
    int tid = threadIdx.x; 
    if (tid > n) { 
       return; 
    } 
    int temp = a[tid]; 
    b[tid] = __all(temp > 100); 
} 
 
__global__ void vote_any(int* a, int* b, int n) 
{ 
    int tid = threadIdx.x; 
    if (tid > n) { 
       return; 
    } 
    int temp = a[tid]; 
    b[tid] = __any(temp > 100); 
}
```



