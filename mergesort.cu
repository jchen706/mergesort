#include "cuda_runtime.h"
#include <stdio.h> 
#include <iostream>
#include <cooperative_groups.h>
#include <algorithm>

/**
 * odd even sort for each tile
 * merge tiles in a block
 * binary search entire grid for the global index of the number
 * 
 * /

/**
 * @brief 
 * 
 * To run the program: ./a.out <number of elements> <random seed number>
 * $ nvcc mergesort 
 * $ ./a.out 100 1
 * $ ./a.out 10000 1
 * 
**/

#define THREADS 1024

namespace cg = cooperative_groups;
__global__ void merge_sort(int* a, int* b, int size, int tile)
{
   auto g = cg::this_grid();

   int grid = gridDim.x;
   
   int threadId = threadIdx.x;
   int blockdim = blockDim.x;
   int blockSize = blockDim.x * tile;
   int my_index = blockIdx.x * blockSize + threadIdx.x * tile;
   int start_of_block = blockIdx.x * blockSize;

   // length of nums for this thread worst case scenario
   int thread_block_size = blockSize;
   if(size < blockSize) {
       thread_block_size = size;
   } else if(size - (start_of_block ) < blockSize) {
       thread_block_size = size - start_of_block;
   }

   int thread_tile_size = tile;
   int start_of_thread_tile = my_index;
   if(size - start_of_thread_tile < tile) {
       thread_tile_size = size - start_of_thread_tile;
   }

     //, sort current tile with odd even sort
     for(int i = 0; i<thread_tile_size; i++) {
       for (int j = 0; j<thread_tile_size-1; j+=2) {
              // index
              int index = my_index + j;
              if(a[index] > a[index+1]) {
                     int temp = a[index+1];
                     a[index+1] = a[index];
                     a[index] = temp;
              }
       }
        
       // check odds
       for (int j = 1; j<thread_tile_size-1; j+=2) {
              int index = my_index + j;
              if(a[index] > a[index+1]) {
                 int temp = a[index+1];
                 a[index+1] = a[index];
                 a[index] = temp;
              }
       }
      }
      __syncthreads();

  // merge sort variation
  for(int k = 2; k <= blockSize; k*=2) {
       
       if(threadId %  k == 0 && threadId < thread_block_size) {
              int start = my_index;
              int segment_size = (k/2) * tile;

              int start2 = my_index + segment_size;
              if(start2 >= start_of_block + thread_block_size) {
                     // start overlaps to the next block
                     // printf(" larger tid: %d k: %d  , %d\n", threadId, k, start2 - start_of_block );
              } else {
                     // size of the next tile 
                     int size2 = (start_of_block + thread_block_size) - start2 < segment_size ? (start_of_block +thread_block_size) - start2 : segment_size;
                     int curr_index = start;
                  
                     int t_start = start;
                     int t_start2 = start2;

                     while(t_start < start+segment_size && t_start2 < start2 + size2) {
                            if(a[t_start] >= a[t_start2]) {
                                   b[curr_index] = a[t_start2];
                                   curr_index += 1;
                                   t_start2 += 1;
                            } else {
                                   b[curr_index] = a[t_start];
                                   t_start+=1;
                                   curr_index +=1;
                            }

                     }
                     
                     while(t_start < start+segment_size) {
                            b[curr_index] = a[t_start];
                            t_start+=1;
                            curr_index +=1;
                     }
                     while(t_start2 < start2 + size2){
                            b[curr_index] = a[t_start2];
                            curr_index += 1;
                            t_start2 += 1;
                     }

                     int s = my_index;
                     for(int m = start; s < start2 + size2; s++){
                            a[s] = b[s];
                     }     
              }
       }
       __syncthreads();
  
  }
//        if(blockIdx.x == 0 && threadIdx.x == 0) {
//               // printf("\n k: %d checking in shared \n", k);
//               // for (int j = 0; j< size; j++) {
//               //        printf(" %d ", b[j]);
//               // }
//               // printf("\n");
//               printf("\n checking in gpu 2nd\n");
//               for (int j = 0; j< size; j++) {
//                      if (j % TILE == 0 ) {
//                             printf("\n  %d  %d A: ",TILE, THREADS);
//                      }
//                      printf(" %d ", a[j]);
                    
//                      // if (j % (TILE*THREADS) == 0) {
//                      //        printf("\n new \n");
//                      // }
//               }
//               printf("\n");

// //        // check decreasing
//       }

  g.sync();

  for(int l = 0; l< thread_tile_size; l++) {
       int current_index = my_index + l;
       
       int numbers_before = current_index - start_of_block; 
       if(current_index < size) {
           
              int curr = a[current_index];
              for(int k = 0; k<grid; k++) {
                     if(k == blockIdx.x) {
                            continue;
                     } 
                     int kstart = k * blockSize;
                     int sub_start = kstart;
                     int end = kstart + blockSize - 1;

                     if(size - kstart < blockSize) {
                         end = kstart + (size - kstart) - 1;
                     }
                     int end_block = end;
                   
                    

                     // if(blockIdx.x == 1 && curr ==  7229) {
                     //        printf("kstart end  %d , %d ,  %d , b: %d kstartnum: %d  nb: %d, c: %d, st: %d\n", kstart, end, curr, blockIdx.x, a[kstart], numbers_before, current_index, start_of_block);
                     // }
                     if ((k > blockIdx.x && a[kstart] == curr) || a[kstart] > curr ) {
                            continue;
                     }

                     if((k < blockIdx.x && curr == a[end]) || a[end] < curr) {
                            numbers_before += (end - sub_start + 1);
                            continue;
                     }


                    
              
                     while(end >= kstart) {
                            int mid = kstart + (end - kstart) / 2;
                            
                            if (a[mid+1] > curr && a[mid] < curr) {
                                         
                                   numbers_before += (mid - sub_start) + 1;
                                   break;
                            } else if (k < blockIdx.x && a[mid] <= curr && a[mid+1] > curr) {
                                   
                                   numbers_before += (mid - sub_start) + 1;
                                   break;
                            } else if (k > blockIdx.x && a[mid] < curr && a[mid+1] >= curr) {
                                   // if(blockIdx.x == 1 && curr ==  7229) {
                                   //        printf(" hereee 2  kstart  %d , end %d ,  current-num %d , b: %d  mid: %d midnum: %d  nb: %d, current-index: %d, st: %d\n", kstart, end, curr, blockIdx.x, mid, a[mid], numbers_before, current_index, start_of_block);
                                   // }  
                                  
                                   numbers_before += (mid - sub_start) + 1;
                                  
                                   break;
                            } else if(a[mid] == curr) {
                                   if(blockIdx.x > k) {
                                          while(mid <= end_block && a[mid] <= curr) {
                                                 mid+=1;
                                                 if(a[mid] > curr) {
                                                        break;
                                                 }
                                          }
                                          numbers_before += (mid - sub_start);
                                          
                                          break;
                                   } else {
                                          while(mid >= sub_start && a[mid] >= curr) {
                                                 mid-=1;
                                                 if(a[mid] < curr) {
                                                        break;
                                                 }
                                                 
                                          }
                                          numbers_before += (mid - sub_start) + 1;
                                         
                                          break;
                                   }
                            }
                            else if(a[mid] < curr) {
                                   kstart = mid + 1;
                            } else {
                                   end = mid -1;
                            }
                     }
                     
              }
             

              // if(blockIdx.x == 1 && curr == 5705) {
              //        printf("curr %d , numbers-before, %d added %d\n", curr, numbers_before, added);
              // }
              b[numbers_before] = curr;
               
              
       }
        
  }
//    if(blockIdx.x == 2 && threadIdx.x == 0) {
//        for(int i = 0; i< TILE; i++) {
//          printf(" nums: %d ", location[i]);
//        }
//     }
 //}
}

int main(int argc, char* argv[]) {

  // generate random numbers 
  printf("Check Cuda Device\n");

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Number of multiprocessors for device 0: %d\n", prop.multiProcessorCount);

  int maxCooperativeBlocks = prop.multiProcessorCount;

  if(argc < 3) {
       std::cerr << "Usage: " << argv[0] << " <SizeOfArray1]> " << " <RandomGeneratorSeed>"  << std::endl;
       std::cerr << "Example: ./a.out 10 1"  << std::endl;
       std::cerr << "Example: ./a.out 100 1"  << std::endl;
       std::cerr << "Example: ./a.out 1000 1"  << std::endl;
       return 1;
  }

  int array1Size  = strtol(argv[1], NULL, 10);
  int randSeed  = strtol(argv[2], NULL, 10);
  srand(randSeed);
//   int array1Size  = 10;

  int array2Size = array1Size;
  printf("Generate Random Numbers with input size of %d and output size of %d\n", array1Size, array2Size);

  int *array1 = new int[array1Size];
  int *array2 = new int[array2Size];
  int *testArray = new int[array1Size];

  int larger = (array1Size > array2Size) ? array1Size : array2Size;
  for(int i = 0; i < larger; i++) {
         if(i < array1Size) {
              array1[i] = rand();
              testArray[i] = array1[i];
         }
         if(i < array2Size) {
              array2[i] = 0;
         }
  }

  int *a_device;
  int *b_device;
  /*
       Determine blocks and threads 
       max number of elements = max cooperative blocks * thread * tile
       2  4  80
       if (num > 80 * threads * 4 && threads <= 1024) {
              threads*=2 
       }
       if (num > 80 * threads * 4) {
              tiles*=2 
       }
  */
  int baseTILE = 4;
  int baseTHREAD = 4;

  // 4 16 32 128 256 512 1024
//   while(baseTHREAD < 1024 && larger > (maxCooperativeBlocks * baseTHREAD * baseTILE) ) {
//          baseTHREAD*=2;
//   }
  while(larger > (maxCooperativeBlocks * 1024 * baseTILE) ) {
         baseTILE*=2;
  }
//   printf(" baseThread: %d baseTile: %d, max number of elements: %d \n", baseTHREAD, baseTILE, maxCooperativeBlocks * baseTHREAD * baseTILE);

//   //   exit(0);


  printf(" Thread: %d Tile: %d, max number of elements: %d \n", THREADS, baseTILE, maxCooperativeBlocks * THREADS * baseTILE);

  int numBlocks = (larger / (THREADS * baseTILE));
  printf(" mod %d \n", (larger % (THREADS * baseTILE)));
  if(larger % (THREADS * baseTILE) > 0) {
       numBlocks += 1;
  }

  int threads = THREADS;

  // Allocate memory on the device
  int array1_byte_size = array1Size * sizeof(int);
  int array2_byte_size = array2Size * sizeof(int);

  cudaMalloc(&a_device, array1_byte_size);
  cudaMalloc(&b_device, array2_byte_size);

  // Copy the input arrays to the device
  cudaMemcpy(a_device, array1, array1_byte_size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_device, array2, array2_byte_size, cudaMemcpyHostToDevice);


  printf("numBlocks: %d \n", numBlocks);
  void *args[] = {&a_device, &b_device, &larger, &baseTILE};

  cudaError_t cudaerr1 = cudaLaunchCooperativeKernel((void*)merge_sort, dim3(numBlocks), dim3(threads), args, 0, NULL);

  if (cudaerr1 != cudaSuccess) {
       printf("kernel launch failed with error \"%s\".\n",
                     cudaGetErrorString(cudaerr1));
  }
     
  //   merge_sort<<<numBlocks, threads>>>(a_device, b_device);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
       printf("kernel launch failed with error \"%s\".\n",
                     cudaGetErrorString(cudaerr));
  }
              
  int * a_out = (int*) malloc(array1_byte_size);

  int * b_out = (int*) malloc(array2_byte_size);

  cudaMemcpy(a_out, a_device, array1_byte_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(b_out, b_device, array2_byte_size, cudaMemcpyDeviceToHost);

  // check the outputs

  std::sort(testArray, testArray + larger);

// printf("\n print output array \n");
//   for(int i = 0; i< larger; i++) {
//        printf(" %d ", b_out[i]);
//   }
//   printf("\n");
  printf("\n Checking Output Array \n");
  for(int i = 0; i< larger; i++) {
       if(b_out[i] != testArray[i]) {
              printf(" \n Incorrect index: %d num: %d actual: %d\n", i, b_out[i], testArray[i]);
              // break;
       }
  }
  printf("\n");
  printf("Finish\n");
  cudaFree(a_device); cudaFree(b_device);
  cudaDeviceReset();
}




















