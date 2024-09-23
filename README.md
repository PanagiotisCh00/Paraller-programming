# Parallel K-means Clustering

Project Overview

This project involves parallelizing the K-means clustering algorithm using different parallel programming techniques. The original K-means algorithm, which classifies points in a 2D space into clusters based on their position, was provided. The project focuses on implementing parallel versions of the K-means algorithm using Pthreads, OpenMP, CUDA, and SIMD to optimize performance and reduce execution time.
Parallelization Techniques

    Pthreads:
        Implemented using static and dynamic workload distribution across threads.
        Static Assignment (lloyd_pThreads_static): Work is evenly divided among threads based on the number of points.
        Dynamic Assignment (lloyd_pThreads_dynamic): Threads dynamically take work from the pool to balance the load.

    OpenMP:
        Static (lloyd_omp_static): Parallel regions are statically assigned to threads.
        Dynamic (lloyd_omp_dynamic): Dynamic scheduling allows load balancing during runtime.
        Guided (lloyd_omp_guided): Similar to dynamic scheduling but with decreasing chunk sizes.

    CUDA:
        Implemented to leverage the GPU for parallelizing the computations. Using CUDA, the K-means algorithm is optimized to run on the NVIDIA GeForce GTX 750 Ti.
        Profiling: Utilized the NVIDIA Visual Profiler to analyze GPU performance and identify bottlenecks.

    SIMD:
        The algorithm was also optimized using SIMD (Single Instruction, Multiple Data) to achieve data-level parallelism.

Performance Analysis

The parallel implementations were benchmarked against the serial version to measure speedup and efficiency across different systems:

    Pthreads and OpenMP were tested on the University’s HPC systems to find the optimal number of threads for the problem.
    CUDA was tested on lab machines with the NVIDIA GTX 750 Ti GPU, using profiling tools to analyze execution times.
    SIMD provided a further optimization by enabling vectorized operations.

Results and Observations

The results, including speedup graphs and efficiency analysis, are documented and discussed based on experiments with the university's HPC and lab systems. Key findings include:

    Optimal thread counts for different parallelization methods.
    A comparison between static, dynamic, and guided workloads.
    Insights into GPU vs. CPU performance, highlighting where each excels and potential bottlenecks.

This project was developed as part of the Parallel Processing course (EPL325) at the University of Cyprus, under the supervision of Prof. Sazeides Yanos.
