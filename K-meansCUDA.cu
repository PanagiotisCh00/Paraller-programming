/* How to Run
Compile Using:
  gcc -Werror -Wall -O3 -lm K-meansSerial_std.c
Run Using:
	./a.out [NumberOfPoints NumberOfClusters]
	./a.out 100000 1024
For gprof:
	gcc -Werror -Wall -pg -lm K-meansSerial_std.c
	./a.out
	gprof ./a.out > analysis.txt
	gprof ./a.out | ./gprof2dot.py | dot -Tpng -o output.png
For perf:
	 perf record -g -- ./a.out
	 perf script | c++filt | ./gprof2dot.py -f perf | dot -Tpng -o output.png
eps-to-jpg
  https://cloudconvert.com/eps-to-jpg
Reference:
 https://github.com/jrfonseca/gprof2dot
 gcc -fopenmp -Wall -Werror -lm -O3 K-meansSerial_std.c

nvcc -ccbin g++ -I../../Common -lm -m64 -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets -O3  K-meansCUDA1013138.cu
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "support.h"

#include <omp.h>

#define NUMBER_OF_POINTS 100000
#define NUMBER_OF_CLUSTERS 20
#define MAXIMUM_ITERATIONS 100
#define SIMD_WIDTH 256
#define SIMD_STEP 256 / 64

// 100 000

// This Version is configured for GTX750Ti
// Consider one Block per SM
// kame ta blocka 1,2,4,5,10,20
#define NUMBER_BLOCKS 50
#define NUMBER_THREADS_PER_BLOCK 1000

void AssignInitialClusteringRandomly(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i; /*Assign initial clustering randomly using the Random Partition method*/
	for (i = 0; i < num_pts; i++)
		pts_group[i] = i % num_clusters;
}
void InitializeClusterXY(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i;
	for (i = 0; i < num_clusters; i++)
	{
		centroids_group[i] = 0; /* used to count the cluster members. */
		centroids[0][i] = 0;	/* used for x value totals. */
		centroids[1][i] = 0;	/* used for y value totals. */
	}
}
void AddToClusterTotal(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i, clusterIndex;
	for (i = 0; i < num_pts; i++)
	{
		clusterIndex = pts_group[i];
		centroids_group[clusterIndex]++;
		centroids[0][clusterIndex] += pts[0][i];
		centroids[1][clusterIndex] += pts[1][i];
	}
}
void DivideEachClusterTotal(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i;
	for (i = 0; i < num_clusters; i++)
	{
		centroids[0][i] /= centroids_group[i];
		centroids[1][i] /= centroids_group[i];
	}
}
/*------------------------------------------------------------------
	findCentrKernel  in kernel.
	This function executes in Kernel. At first i calculate the unique id of each thread based on the block and thread is belonged and the block it belongs.
	Then i calculate where its iterations start and finish like we saw in the labs.
	I "parralelized" the outer for because there are not any dependencies among the iterations like we did in the previous exercises.
	Each thread calculates the changes it made and adds it to the array changesArr in the position of its threadIDx. It waits for synchronization  and
	then the thread with id=0 of each block adds up all the changes made by every thread of the block in the array dchanges in the position blockID,
	and in every position of dchanges there is the changes made by the block that is its index.
	The number of blocks and the number of threads of each block are global variables in the program.

---------------------------------------------------------------------*/
__global__ void findCentrKernel(double *dCentroids0, double *dCentroids1, double *dpts0, double *dpts1, unsigned int *dptsgroup, int num_pts, int num_clusters, int *dchanges, int *changesArr)
{
	//  Calculate the Unique Thread ID
	int threadUniqueID = threadIdx.x + blockIdx.x * blockDim.x;
	// int threadUniqueID = threadIdx.x;
	//  How Many iterations a thread should perform *correct *
	int threadNumberOfIterations = (num_pts / gridDim.x) / blockDim.x;
	// Which is the first element for this thread
	int threadStartFrom = threadUniqueID * (threadNumberOfIterations);
	int i, j, clusterIndex = 0;
	double x, y, d, min_d;
	int blockID = blockIdx.x;

	for (i = threadStartFrom; i < threadStartFrom + threadNumberOfIterations; i++)
	{
		min_d = 0x7f800000; // IEEE 754 +infinity
		for (j = 0; j < num_clusters; j++)
		{
			x = dCentroids0[j] - dpts0[i];
			y = dCentroids1[j] - dpts1[i];
			d = x * x + y * y;
			if (d < min_d)
			{
				min_d = d;
				clusterIndex = j;
			}
		}
		if (clusterIndex != dptsgroup[i])
		{
			dptsgroup[i] = clusterIndex;
			changesArr[threadIdx.x]++;
		}
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < NUMBER_THREADS_PER_BLOCK; i++)
		{
			dchanges[blockID] += changesArr[i];
		}
	}
}

/*------------------------------------------------------------------
	FindNearestCentroidCUDA cuda version
	I have global variables about the number of blocks and the threads that each block will have.
	I used unified memory to solve this problem so i would have better results, in main i created the variables that i use in kernel using cudaMallocManged,
	and here i created a variable that will hold the result of the changes made by each block and i add them up in here with the cpu.
	I also have an array with the size of number of blocks * number of threads in each block which will hold the number of changes every thread of each block made.
	I create a variable that has the number of blocks of the global variable and another one that has the number of threads per block of the global variable.
	I call the kernel function findCentrKernel with all the parameters, i send them as 1-D array the centroids ands points and the total changes of each block
	are in the variable dchanges. Then i use the cudaDevice synchronize to synchronize all the threads, blocks of the gpu and add up all the changes made by each
	block in the variable changes and return it.

---------------------------------------------------------------------*/
int FindNearestCentroidCUDA(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int *changes = (int *)malloc(sizeof(int));
	int *changesBlocks = (int *)malloc(sizeof(int) * NUMBER_BLOCKS);
	int *dChanges;
	int *dchangesArr;

	cudaMallocManaged((void **)&dChanges, NUMBER_BLOCKS * sizeof(int));
	cudaMalloc((void **)&dchangesArr, NUMBER_THREADS_PER_BLOCK * NUMBER_BLOCKS * sizeof(int));

	// kernel invocation code
	dim3 dimBlock(NUMBER_THREADS_PER_BLOCK);
	dim3 dimGrid(NUMBER_BLOCKS);
	findCentrKernel<<<dimGrid, dimBlock>>>(centroids[0], centroids[1], pts[0], pts[1], pts_group, num_pts, num_clusters, dChanges, dchangesArr);
	cudaDeviceSynchronize();
	int i;
	for (i = 0; i < NUMBER_BLOCKS; i++)
	{
		*changes += dChanges[i];
	}
	return *changes;
}

int FindNearestCentroid(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i, j, clusterIndex = 0;
	int changes = 0;
	double x, y, d, min_d;

	for (i = 0; i < num_pts; i++)
	{
		min_d = 0x7f800000; // IEEE 754 +infinity
		for (j = 0; j < num_clusters; j++)
		{
			x = centroids[0][j] - pts[0][i];
			y = centroids[1][j] - pts[1][i];
			d = x * x + y * y;
			if (d < min_d)
			{
				min_d = d;
				clusterIndex = j;
			}
		}
		if (clusterIndex != pts_group[i])
		{
			pts_group[i] = clusterIndex;
			changes++;
		}
	}
	return changes;
}

/*------------------------------------------------------------------
	lloyd
	This function clusters the data using Lloyd's K-Means algorithm
	after selecting the intial centroids.
	It returns a pointer to the memory it allocates containing
	the array of cluster centroids.
---------------------------------------------------------------------*/
void lloyd(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i;
	int changes;
	int acceptable = num_pts / 1000; /* The maximum point changes acceptable. */

	if (num_clusters == 1 || num_pts <= 0 || num_clusters > num_pts)
		return;

	if (maxTimes < 1)
		maxTimes = 1;
	/*Assign initial clustering randomly using the Random Partition method*/
	AssignInitialClusteringRandomly(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

	do
	{
		/* Calculate the centroid of each cluster.
		  ----------------------------------------*/
		/* Initialize the x, y and cluster totals. */
		InitializeClusterXY(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Add each observation's x and y to its cluster total. */
		AddToClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Divide each cluster's x and y totals by its number of data points. */
		DivideEachClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Find each data point's nearest centroid */
		changes = FindNearestCentroid(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

		;
		maxTimes--;
	} while ((changes > acceptable) && (maxTimes > 0));

	/* Set each centroid's group index */
	for (i = 0; i < num_clusters; i++)
		centroids_group[i] = i;
} /* end, lloyd */

/*------------------------------------------------------------------
	lloyd cuda
	This is the cuda function-version of lloyd, which gets as a parameter the number of threads and call the function FindNearestCentroidCUDA.
	which is parralelized using cuda
	This function clusters the data using Lloyd's K-Means algorithm
	after selecting the intial centroids.
	It returns a pointer to the memory it allocates containing
	the array of cluster centroids.
---------------------------------------------------------------------*/
void lloyd_CUDA(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i;
	int changes;
	int acceptable = num_pts / 1000; /* The maximum point changes acceptable. */

	if (num_clusters == 1 || num_pts <= 0 || num_clusters > num_pts)
		return;

	if (maxTimes < 1)
		maxTimes = 1;
	/*Assign initial clustering randomly using the Random Partition method*/
	AssignInitialClusteringRandomly(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

	do
	{
		/* Calculate the centroid of each cluster.
		  ----------------------------------------*/
		/* Initialize the x, y and cluster totals. */
		InitializeClusterXY(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Add each observation's x and y to its cluster total. */
		AddToClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Divide each cluster's x and y totals by its number of data points. */
		DivideEachClusterTotal(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		/* Find each data point's nearest centroid */
		// changes = FindNearestCentroid(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
		changes = FindNearestCentroidCUDA(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

		;
		maxTimes--;
	} while ((changes > acceptable) && (maxTimes > 0));

	/* Set each centroid's group index */
	for (i = 0; i < num_clusters; i++)
		centroids_group[i] = i;
}

void print_centroids_v3(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters)
{
	int i;
	for (i = 0; i < num_pts; i++)
		centroids_group[(int)pts_group[i]]++;
	for (i = 0; i < num_clusters; i++)
		printf("\n%% Group:%d #ofPoints %d:\t\tcentroids.x:%f\tcentroids.y:%f", i, centroids_group[i], centroids[0][i], centroids[1][i]);
	printf("\n");
}
/*-------------------------------------------------------
	main
-------------------------------------------------------*/
int main(int argc, char **argv)
{
	int num_pts = NUMBER_OF_POINTS;
	int num_clusters = NUMBER_OF_CLUSTERS;
	int maxTimes = MAXIMUM_ITERATIONS;
	int i, nrows = 2;
	double radius = RADIUS;
	double **pts;
	unsigned int *pts_group;
	double **centroids;
	unsigned int *centroids_group;
	int numThreads;
	if (argc == 3)
	{
		num_pts = atoi(argv[1]);
		num_clusters = atoi(argv[2]);
		numThreads = 1;
	}
	else if (argc == 4)
	{
		num_pts = atoi(argv[1]);
		num_clusters = atoi(argv[2]);
		numThreads = atoi(argv[3]);
	}
	else if (argc == 2)
	{
		printf("%%*** RUNNING WITH DEFAULT VALUES ***\n");
		printf("%%Execution: ./a.out Number_of_Points Number_of_Clusters\n");
		numThreads = atoi(argv[1]);
	}

	else
	{
		printf("%%*** RUNNING WITH DEFAULT VALUES ***\n");
		printf("%%Execution: ./a.out Number_of_Points Number_of_Clusters\n");
		numThreads = 1;
	}
	printf("Number of threads %d\n", numThreads);
	printf("%%Number of Points:%d, Number of Clusters:%d, maxTimes:%d,radious:%4.2f\n", num_pts, num_clusters, maxTimes, radius);
	/* Generate the observations */
	// printf("%%SERIAL: Kmeans_initData\n");
	pts = Kmeans_initData_v3(num_pts, radius);
	// pts_group = (unsigned int *)malloc(num_pts * sizeof(unsigned int));
	cudaMallocManaged(&pts_group, num_pts * sizeof(unsigned int));

	// centroids = (double **)malloc(nrows * sizeof(double *));
	cudaMallocManaged(&centroids, nrows * sizeof(double *));

	for (i = 0; i < nrows; i++)
		// centroids[i] = (double *)malloc(num_clusters * sizeof(double));
		cudaMallocManaged(&centroids[i], num_clusters * sizeof(double));

	centroids_group = (unsigned int *)malloc(num_clusters * sizeof(unsigned int));
	startTime(0);
	/* Cluster using the Lloyd algorithm and K-Means++ initial centroids. */

	// lloyd(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
	lloyd_CUDA(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

	stopTime(0);
	/* Print the results */
	// print_eps_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters);

	print_centroids_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters);
	elapsedTime(0);
	cudaFree(pts);
	cudaFree(centroids);
	return 0;
}
