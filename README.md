# Copyright (c) Micu Florian-Luis 331CA 2022 - Assignment 3

# Motivation
This assignment's purpose is to familiarize one with hardware acceleration, 
more precisely GPU programming with CUDA. The task at hand is to find all of 
the cities in range of a city and add all of their populations. 

To complete this assignment, vast research was needed for various GPU 
optimization techniques whilst also understanding how the GPU stores and 
operates with its data. 

# Implementation
## CPU Only
First, I had to complete the main task on the CPU. To do this, I read from the
input file all of the cities' data and then I added two for loops so that each
city verifies the other cities distances. I first had to extract all of the 
data which meant I could not do any computations before I read the file 
completely. This initial implementation worked, however there is a significant
optimization to be done, specifically when a valid city is found the program
shall add both of the cities populations to each others counters. This means
that the outer for loop remains the same (range [0, N)), however the inner for
loop will only go from the current city to the last city (range [i, N)), thus
increasing performance (by half from my personal tests). Moreover, in 
reading part, the total population count could be already initialized to be
equal to its population. With these said the algorithm evolved like this:

Before:
for i in 0...N
    for j in 0...N

After:
for i in 0...N
    for j in i+1...N

## Note
1. I could calculate partial results whilst reading the data, however that will
complicate the step of adding GPU code. Furthermore, not much of a difference
in time was found with this idea.
2. The strings that hold the names of the cities are not used for anything,
however I let them in the code so that no data is lost.

## CPU + GPU
Second, to pass the tests in due time, I had to implement a kernel function.
For this, I used Host allocated memory to store the read data from the input
file and then I allocated adequate memory on the Device so that I could copy
the read data. In the kernel function, I let each thread have their own city
and they run the inner for loop finding cities that are in range. I tried
running with a 2D block to eliminate the for loop, however it ran slower
compared to leaving the for loop there (L1-L2 caching might have been the
saviour). After the threads finish, the output data is copied back to the
Host to be printed.

## Note
1. Block size was chosen by trial-and-error with sizes such as 256, 512 and
1024. The conclusion was that size 256 runs the best.
2. Directives for GPU memory where not needed since the compiler will
automatically assign the constant values in the thread register. Moreover,
to do less computations, pointers were used for constant data.
3. Since there might be a race condition when threads are adding values,
atomic operations are needed (atomicAdd).
4. The geoDistance function must be made callable from the GPU, therefore I
added the prefix "__device__".

# Performance
This algorithm performers better than a plain CPU algorithm, however it does
not manage to pass the last test H1. Without that test, it manages on average
a performance of 10s on K40M which is below the maximum of 30s. With the H1
test, the program finishes on average in 1 minute and 20s. Since the program
is now multithreaded it was expected to perform better than before.

Some ways in which the program could be further improved are:
1. K-d Trees: this data structure is used specifically for rangesearch queries,
however its implementation is not easy and porting it to the GPU may be harder.
I tried using a 2D K-d Tree from GitHub, however it had a lot of boilerplate
and it ran slower.
2. Not calling geoDistance: it is possible to just translate the longitude and
latitude to kilometers and perform a simple if check which will arguably 
perform much better. This solution is used in some SQL queries, however it is
not applicable here since we must not change the complexity of the geoDistance
algorithm.
3. Using CUDA streams: NVidia implemented streams so that kernels could run
whilst data is copied which would definitely improve performance as we must
first read data from files that have 1M entries. I tried implementing this, 
however it did not work correctly as it requires asynchronism, thus becoming
harder to implement for this specific task.
4. Transforming geoDistance in a child Kernel: this function could benefit
from becoming itself a kernel, however from my tests the output was not
correct anymore, therefore more research is needed.

## Note
1. To see the performance of this algorithm, I added two screenshots from fep
running in K40M with and without the H1 test. Both tests were done using the
executable time directly on the CUDA program, not on the fep script.
2. I want my assignment to be graded without the H1 test (thus I will have 70p
out of 90p from the tests), hence why I made the for loop inside "main" go only
to "argc - 3".

# Feedback
Cool homework, however I am not sure what is required to pass all tests, which
is why I cannot say that I enjoyed it too much :( Probably more in depth CUDA
examples on OCW would have helped.

# Bibliography
Blocks size: http://www.mathcs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/2D-grids.html
SQL Queries optimization: http://www.plumislandmedia.net/mysql/haversine-mysql-nearest-loc/
CUDA Optimization Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
CUDA Streams: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation__concurrent-copy-and-execute
K-d Trees: https://www.youtube.com/watch?v=BK5x7IUTIyU
MATLAB's own rangesearch using K-d Trees: https://www.mathworks.com/help/stats/rangesearch.html
OCW labs 7-8-9
