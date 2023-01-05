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

  gcc -fopenmp -Wall -lm K-meansSerial_std.c
 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "support.h"

#include <pthread.h>

#define NUMBER_OF_POINTS 100000
#define NUMBER_OF_CLUSTERS 20
#define MAXIMUM_ITERATIONS 100
#define SIMD_WIDTH 256
#define SIMD_STEP 256 / 64

///#define CONST_CHUNKSIZE 100

/*------------------------------------------------------------------
This is the struct that is used for the static and dynamic schedule, to save all the information and data a thread needs to be executed correctly.
It contains the id of the thread, the total number of threads, the array of the points, the number of the clusters, the number of the points, the
number of changes that were made by that thread, the array with the pts_group and the size of the chunkSize.

---------------------------------------------------------------------*/
typedef struct
{
	int id;
	int numThreads;
	double **centroids;
	double **pts;
	int num_clusters;
	int num_pts;
	int changes;
	unsigned int *pts_group;
	int chunk_size;

} package_t;
static int chunk_start = 0;

int arraySizeGLOBAL = 0;

static int chunkSize;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

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
	getNextChunkStart
	This function gives me the next chunk Start for the threads. It calculates
	based on the chunksize (which is a global variable equal to number of points/number of threads) where the next thread should start
	its iterations. First it locks the mutex, because this function can be used by many threads at same time but only 1 thread can access
	chunk start at any moment so it locks the mutex makes the addition(chunk_start += chunkSize) unlocks the mutex and returns the
	chunks start and if its bigger than the size of the array that is being parallelized then returns -1 so that thread will die(exit) in the
	process that calls the getNextChunk start.
---------------------------------------------------------------------*/
int getNextChunkStart()
{
	/*It's better to lock in getNextChunkStart()*/
	pthread_mutex_lock(&mutex);
	// chunk_start += CONST_CHUNKSIZE;
	chunk_start += chunkSize;
	pthread_mutex_unlock(&mutex);
	return ((chunk_start < arraySizeGLOBAL) ? chunk_start : -1);
}

/*------------------------------------------------------------------
	lloyd FindNearestCentroidDynamicWorker (Dynamic thread function)
	This function get as a parameter a struct with all the information, data, a thread needs so every thread will have access to the right information to write and read
	as i explained before.
	I decided to parallelize the first for loop(the outer) because every iteration does not have any dependency to any previous iteration. I decided not to
	parallelize the inner for loop because it would have dependency problem with d and min_d  and it would produce wrong results, and if i corrected it it would make
	the program really slow. The number of the changes that the thread made, will return it using the struct as i make the changes in that.
	The iteration start for the thread is calculated with the function getNextChunkStart which gives the next available chunk start based on the chunk size
	which is number of point/number of threads. The function getNextChunkStart will return a negative chunk start if there is no other iteration of the array
	so in that case the thread will exit.
	This paraller function has Dynamic schedule with chunkSize equals to num of points/ num of threads.
---------------------------------------------------------------------*/
void *FindNearestCentroidDynamicWorker(void *arg)
{
	package_t *p = (package_t *)arg;
	int i, j, start, clusterIndex = 0;
	double x, y, d, min_d;
	while (1)
	{
		start = getNextChunkStart();
		if (start < 0)
		{
			// stopTime(p->id);
			pthread_exit(NULL);
		}
		p->chunk_size += 1;
		for (i = start; (i < start + chunkSize) && (i < p->num_pts); i++)
		{
			min_d = 0x7f800000; // IEEE 754 +infinity
			for (j = 0; j < p->num_clusters; j++)
			{
				x = p->centroids[0][j] - p->pts[0][i];
				y = p->centroids[1][j] - p->pts[1][i];
				d = x * x + y * y;
				if (d < min_d)
				{
					min_d = d;
					clusterIndex = j;
				}
			}
			if (clusterIndex != p->pts_group[i])
			{
				p->pts_group[i] = clusterIndex;
				p->changes++;
			}
		}
	}
	pthread_exit(NULL);
}
/*------------------------------------------------------------------
	lloyd FindNearestCentroidStaticWorker (static thread function)
	This function get as a parameter a struct with all the information, data, a thread needs so every thread will have access to the right information to write and read
	as i explained before.
	I decided to parallelize the first for loop(the outer) because every iteration does not have any dependency to any previous iteration. I decided not to
	parallelize the inner for loop because it would have dependency problem with d and min_d  and it would produce wrong results, and if i corrected it it would make
	the program really slow. The number of the changes that the thread made, will return it using the struct as i make the changes in that.
	The iterations for the thread starts at chunk size * id of the thread and continues untill it reach the chunk Size *(id+1) or until it reaches the end of the
	array(number of points).
	This paraller function has static schedule with chunkSize equals to num of points/ num of threads.
---------------------------------------------------------------------*/
void *FindNearestCentroidStaticWorker(void *arg)
{
	package_t *p = (package_t *)arg;
	int i, j, clusterIndex = 0;
	double x, y, d, min_d;
	// startTime(p->id);
	for (i = p->chunk_size * p->id; i < p->chunk_size * (p->id + 1) && (i < p->num_pts); i++)
	{
		min_d = 0x7f800000; // IEEE 754 +infinity
		for (j = 0; j < p->num_clusters; j++)
		{
			x = p->centroids[0][j] - p->pts[0][i];
			y = p->centroids[1][j] - p->pts[1][i];
			d = x * x + y * y;
			if (d < min_d)
			{
				min_d = d;
				clusterIndex = j;
			}
		}
		if (clusterIndex != p->pts_group[i])
		{
			p->pts_group[i] = clusterIndex;
			p->changes++;
		}
	}
	// stopTime(p->id);
	pthread_exit(NULL);
}
/*------------------------------------------------------------------
	lloyd findNearestCentroid static
	This function gets as a parameter the number of threads that the parraler region will create. First, it creates all the threads and for each thread create an
	object (struct instance) package_t and adds its data. I add the id of the thread, the number of threads, the array of the centroids, the array of the points,
	the number of the clustes, number of points, changes that count how many changes were made for that thread, the array pts_group and the chunk size
	which if the i is equal with number of threads -1 then it is the number of points i * (number of points/ number of threads), else its equal tu num of points/num of threads.
	So i create the structs i explained before and i create all the threads and i call for each one the function FindNearestCentroidStaticWorker which gets as a
	parameter the struct and it makes the actual parallelized job--work for the thread. Then, i kill each thread (with join) and i add each changes (from the struct)
	to the variable changes which i return after all the threads are finished,done.
	This paraller function has static schedule with chunkSize equals to num of points/ num of threads.
---------------------------------------------------------------------*/
int FindNearestCentroid_static(double **pts, unsigned int *pts_group,
							   int num_pts, double **centroids,
							   unsigned int *centroids_group,
							   int num_clusters, int maxTimes, int numThreads)
{
	int i = 0;
	int changes = 0;
	pthread_t threads[numThreads];
	package_t *p[numThreads];
	int chunk_size = num_pts / numThreads;

	for (i = 0; i < numThreads; i++)
	{
		p[i] = (package_t *)malloc(sizeof(package_t));
		p[i]->id = i;
		p[i]->numThreads = numThreads;
		p[i]->centroids = centroids;
		p[i]->pts = pts;
		p[i]->num_clusters = num_clusters;
		p[i]->num_pts = num_pts;
		p[i]->changes = changes;
		p[i]->pts_group = pts_group;
		p[i]->chunk_size = (i == numThreads - 1) ? num_pts - i * chunk_size : chunk_size;

		pthread_create(&threads[i], NULL, FindNearestCentroidStaticWorker, (void *)p[i]);
	}

	for (i = 0; i < numThreads; i++)
	{
		pthread_join(threads[i], NULL);
		// printf("Thread %d: ", i);
		changes += p[i]->changes;
		// elapsedTime(i);
	}
	return changes;
}

/*------------------------------------------------------------------
	lloyd findNearestCentroid Dynamic
	This function gets as a parameter the number of threads that the parraler region will create. First, it creates all the threads and for each thread create an
	object (struct instance) package_t and adds its data. I add the id of the thread, the number of threads, the array of the centroids, the array of the points,
	the number of the clustes, number of points, changes that count how many changes were made for that thread, the array pts_group.
	So i create the structs i explained before and i create all the threads and i call for each one the function FindNearestCentroidSDynamicWorker which gets as a
	parameter the struct and it makes the actual parallelized job--work for the thread. Then, i kill each thread (with join) and i add each changes (from the struct)
	to the variable changes which i return after all the threads are finished,done.
	This paraller function has static schedule with chunkSize equals to num of points/ num of threads.
---------------------------------------------------------------------*/
int FindNearestCentroid_Dynamic(double **pts,
								unsigned int *pts_group, int num_pts, double **centroids,
								unsigned int *centroids_group, int num_clusters, int maxTimes, int numThreads)
{
	int i = 0;
	int changes = 0;
	arraySizeGLOBAL = num_pts;
	chunkSize = num_pts / numThreads;
	// chunkSize = 1000;
	chunk_start = -chunkSize;

	pthread_t threads[numThreads];
	package_t *p[numThreads];
	for (i = 0; i < numThreads; i++)
	{
		p[i] = (package_t *)malloc(sizeof(package_t));
		p[i]->id = i;
		p[i]->numThreads = numThreads;
		p[i]->centroids = centroids;
		p[i]->pts = pts;
		p[i]->num_clusters = num_clusters;
		p[i]->num_pts = num_pts;
		p[i]->changes = changes;
		p[i]->pts_group = pts_group;
		p[i]->chunk_size = 0;

		pthread_create(&threads[i], NULL, FindNearestCentroidDynamicWorker, (void *)p[i]);
	}

	for (i = 0; i < numThreads; i++)
	{
		pthread_join(threads[i], NULL);
		// printf("Thread %d: ", i);
		changes += p[i]->changes;
		// elapsedTime(i);
	}
	return changes;
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
	//	printf("Changes %d \n", changes);
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
	lloyd static
	This is the "parralelized" function-version of lloyd, which gets as a parameter the number of threads and call the function findNearestCentroid_static
	which is the function that does the parallelization.
	This function clusters the data using Lloyd's K-Means algorithm
	after selecting the intial centroids.
	It returns a pointer to the memory it allocates containing
	the array of cluster centroids.
---------------------------------------------------------------------*/
void lloyd_pThreads_static(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int numThreads)
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
		changes = FindNearestCentroid_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, numThreads);
		maxTimes--;
	} while ((changes > acceptable) && (maxTimes > 0));

	/* Set each centroid's group index */
	for (i = 0; i < num_clusters; i++)
		centroids_group[i] = i;
}
/*------------------------------------------------------------------
	lloyd dynamic
	This is the "parralelized" function-version of lloyd, which gets as a parameter the number of threads and call the function findNearestCentroid_dynamic
	which is the function that does the parallelization.
	This function clusters the data using Lloyd's K-Means algorithm
	after selecting the intial centroids.
	It returns a pointer to the memory it allocates containing
	the array of cluster centroids.
---------------------------------------------------------------------*/
void lloyd_pThreads_dynamic(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int numThreads)
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
		changes = FindNearestCentroid_Dynamic(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, numThreads);

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
		// printf("%%*** RUNNING WITH DEFAULT VALUES ***\n");
		// printf("%%Execution: ./a.out Number_of_Points Number_of_Clusters\n");
		numThreads = atoi(argv[1]);
	}

	else
	{
		/// printf("%%*** RUNNING WITH DEFAULT VALUES ***\n");
		// printf("%%Execution: ./a.out Number_of_Points Number_of_Clusters\n");
		numThreads = 1;
	}

	// printf("Number of threads %d\n", numThreads);
	// printf("%%Number of Points:%d, Number of Clusters:%d, maxTimes:%d,radious:%4.2f\n", num_pts, num_clusters, maxTimes, radius);
	/* Generate the observations */
	// printf("%%SERIAL: Kmeans_initData\n");
	pts = Kmeans_initData_v3(num_pts, radius);
	pts_group = malloc(num_pts * sizeof(unsigned int));
	centroids = malloc(nrows * sizeof(double *));
	for (i = 0; i < nrows; i++)
		centroids[i] = malloc(num_clusters * sizeof(double));
	centroids_group = malloc(num_clusters * sizeof(unsigned int));
	startTime(0);
	/* Cluster using the Lloyd algorithm and K-Means++ initial centroids. */

	// lloyd(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);
	arraySizeGLOBAL = num_pts;

	chunkSize = num_pts / numThreads;

	lloyd_pThreads_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, numThreads);

	// lloyd_pThreads_dynamic(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, numThreads);

	stopTime(0);
	/* Print the results */
	// print_eps_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters);

	// print_centroids_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters);
	elapsedTime(0);

	free(pts);
	free(centroids);
	// printf("\n");
	return 0;
}
