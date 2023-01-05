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

module use /share/apps/eb/modules/all
module load GCC/9.3.0
gcc -fopenmp -Wall -lm -mavx512f K-meansOMPSIMD.c

 */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#include "support.h"

#include <omp.h>
#define NUMBER_OF_POINTS 100000
#define NUMBER_OF_CLUSTERS 20
#define MAXIMUM_ITERATIONS 100
#define SIMD_WIDTH 256
#define SIMD_STEP 256 / 64

void AssignInitialClusteringRandomly(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes)
{
	int i; /*Assign initial clustering randomly using the Random Partition method*/
	for (i = 0; i < num_pts; i++)
		pts_group[i] = i % num_clusters;
}
void AssignInitialClusteringRandomly_OMP(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int numThreads)
{
	omp_set_num_threads(numThreads);
	int i;

#pragma omp parallel private(i) shared(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, numThreads)
	{
#pragma omp for schedule(static, num_pts / numThreads)
		/*Assign initial clustering randomly using the Random Partition method*/
		for (i = 0; i < num_pts; i++)
			pts_group[i] = i % num_clusters;
	}
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
/**
 * @brief this function prints a 512 double vector. Used for debugging purposes
 *
 * @param vec the vector it will print.
 */
void printVector_d(__m512d vec)
{
	double tmp[8];
	_mm512_storeu_pd(tmp, vec);

	int i;
	for (i = 0; i < 8; i++)
	{
		printf("%2.2f\t", tmp[i]);
	}
	printf("\n");
}

/**
 * @brief this function prints a 512 integer vector. Used for debugging purposes
 *
 * @param vec the vector it will print
 */
void printVector_i(__m512i vec)
{
	__attribute__((aligned(512))) int64_t tmp[8];
	_mm512_store_epi64(tmp, vec);

	int64_t i;
	for (i = 0; i < 8; i += 1)
	{
		printf("%ld\t", tmp[i]);
	}
	printf("\n");
}
/**
 * @brief Get the Min Element object
 * I have a vector of 8 elements and i want to return the smallest.
 * I can only do comparisons as vectors,so at first i will create the permutated
 * vector which is the same vector in opposite order(1,2,3,4,5,6,7,8) becomes (8,7,6,5,4,3,2,1)
 * i make the comparison, first and second half of vector is same(1,2,3,4,4,3,2,1).
 * I then permutate based on the second immediate (1,2,3,4,4,3,2,1) becomes (3,4,1,2,2,1,4,3),so i have.
 * I find their min (1,2,1,2,2,1,1,2) so i reduced my numbers from 4 to 2.
 * I use the third immediate to create its permutated (2,1,2,1,1,2,2,1) i use min
 * and in every position there is the minimum number (1).
 * I get the element and return it
 *
 * @param vector the integer vector (has 8 elements)
 * @return int returns the minimum number-element of the vector
 */
int getMinElement(__m512i vector)
{
	// imm1 is for the first permutation,imm2 for the second permutation(when the actual vector has size 4),imm3 is for the third permutation
	//(when the actual vector size is 2)
	__m512i imm1 = _mm512_set_epi64(0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7);
	__m512i imm2 = _mm512_set_epi64(0x5, 0x4, 0x7, 0x6, 0x1, 0x0, 0x3, 0x2);
	__m512i imm3 = _mm512_set_epi64(0x6, 0x7, 0x4, 0x5, 0x2, 0x3, 0x0, 0x1);
	// first permutation and use of min function
	__m512i a = _mm512_permutex2var_epi64(vector, imm1, vector); // permutes based on imm
	__m512i min = _mm512_min_epi64(vector, a);
	// second permutation and use of min function
	a = _mm512_permutex2var_epi64(min, imm2, min);
	min = _mm512_min_epi64(min, a);
	// second permutation and use of min function
	a = _mm512_permutex2var_epi64(min, imm3, min); // permutes based on imm
	min = _mm512_min_epi64(min, a);
	// create integer to return
	__attribute__((aligned(512))) int64_t minElement[8];
	_mm512_store_epi64(minElement, min); // convert to array
	return minElement[0];
}
/**
 * @brief Get the index of the min element of the vector. I pass the vector i want to check and the vector indexes, which are pallarel vectors.
 * I return the index of the min number in the vector. I work in the same way as in the function getMinElement but this time i also change the indexes.
 * For every permutated vector i add the indexes as max int and in the end the vector indexes will be full of int max elements except one that will be the
 * index of the smallest number, so i find the min number in the vector with function getMinElement and return it.
 *
 * @param d the vector i want to find its smallest element
 * @param min_indices the indexes of of the numbers in d
 * @return int return the index of the smallest number.
 */
int getMinAndIndex(__m512d d, __m512i min_indices)
{

	// now i find the minimum element of vector d and its index
	// Always the d is my first vector-min and v2 the vector that
	// the comparison happen with
	// actually i will use 3 vectors so their index have to become bigger by 8,i use the increment vector.
	__m512i increment = _mm512_set1_epi64(INT_MAX);
	// the 3 immediates like i explained in the function find min
	__m512i imm1 = _mm512_set_epi64(0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7);
	__m512i imm2 = _mm512_set_epi64(0x5, 0x4, 0x7, 0x6, 0x1, 0x0, 0x3, 0x2);
	__m512i imm3 = _mm512_set_epi64(0x6, 0x7, 0x4, 0x5, 0x2, 0x3, 0x0, 0x1);

	// First permutation and find min
	__m512i indices = _mm512_add_epi64(min_indices, increment);		   // i add the increment so every vector has different indices
	__m512d v2 = _mm512_permutex2var_pd(d, imm1, d);				   // permutes based on the first imm
	__mmask8 mask = _mm512_cmp_pd_mask(d, v2, 0x1);					   // 0x2 stands for the less than comparison
	min_indices = _mm512_mask_blend_epi64(mask, indices, min_indices); // i add the index of the smallest number in the comparison using the mask.
	d = _mm512_mask_blend_pd(mask, v2, d);							   // i recreate d with the minimum numbers between d and v2

	// Second round of permutation and to find min
	indices = _mm512_add_epi64(indices, increment);					   // i make the indexes bigger
	v2 = _mm512_permutex2var_pd(d, imm2, d);						   // permutes based on imm2 to create its permutated vector that the comparison will happend with
	mask = _mm512_cmp_pd_mask(d, v2, 0x1);							   // comparison for less than
	min_indices = _mm512_mask_blend_epi64(mask, indices, min_indices); // i add the indexes of the smallest numbers in the variable, as before
	d = _mm512_mask_blend_pd(mask, v2, d);							   // i find the smallest numbers in the 2 vectors

	// Third round of permutation and to find min
	indices = _mm512_add_epi64(indices, increment); // i make the indices bigger
	v2 = _mm512_permutex2var_pd(d, imm3, d);		// permutes based on imm3 for the final permutation
	mask = _mm512_cmp_pd_mask(d, v2, 0x1);			// create the mask for the comparison of the 2 vectors
	min_indices = _mm512_mask_blend_epi64(mask, indices, min_indices);
	d = _mm512_mask_blend_pd(mask, v2, d);

	// now d is a vector with 8 same elements, and min_indices a vector that contains in a position
	// the index of the minimum element. That index will be for sure the only number that will be between 0-7 so
	// it will obviously be the smallest number, that is why i use a function to find the minimum number in the vector
	// which will be the index of the smallest element.

	return getMinElement(min_indices); // smallest's number index in the vector
}
/*------------------------------------------------------------------
	lloyd FindNearestCentroidSIMD_static static
	This function gets as a parameter the number of threads that the parraler region will create(and creates it with the omp_set_num).
	I decided to parallelize the first for loop(the outer) because every iteration does not have any dependency to any previous iteration. I decided not to
	parallelize the inner for loop because it would have dependency problem with d and min_d  and it would produce wrong results, and if i corrected it it would make
	the program really slow.I decided to make the variables i,j,x,y,d,min_d private as every thread should see its own instance of that variable(eg every thread must see its own
	i and j to produce correct results). I made shared the variables num_pts,pts,pts_group, numThreads,centroids as every thread can see it and access it and all the
	threads should have the same, and the pts_group in which the thread writed data there is no problem because it writes in different index of the array.
	Also the clusterIndex is first private so it can have the value that has before it entered the paraller part and changes is reduction so it can be
	summed every private insant of changes after the paraller region and produce correct results(i decided to use reduction as its an easy , correct and optimized
	way from the omp library).
	This paraller function has static schedule with chunkSize equals to num of points/ num of threads.
	I also added some vectorization in the inner loop, the loop that executes every one thread.
	Before the for loop for the j i make the initializations for the vectors as such p0,p1 for the array pts, i create a minVec which will have in all posiitons
	the min_d and in every loop will have the minimum number of the iterations made so far and the clusterIndexVec is the index for the elements in minVec.
	I do all the calulations as it was before but now in vector. After finding d i have to make the calculations to compare that vector with the vector minVec,
	with the masking as we learnt in the course and i also use that mask to add the data in vector  clusterIndexVec.
	After the while the vector minVec will have the smallest values and clusterIndexVec will have the respectively indeces.
	The i call the function getMinAndIndex to dinf the index of the smallest value, i make the if with ptsgroup[i] and if they are different i add one to the changes
	and put the value of that index in the ptsgroup[i].
	The function getMinAndIndex is explained later in my report.
---------------------------------------------------------------------*/
int fl = 0;
int flMpr = 0;
int FindNearestCentroidSIMD_static(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int numThreads)
{

	int i, j, clusterIndex = 0;
	int changes = 0;
	// double x, y, d, min_d;
	double min_d;
	__m512d x, y, d;
	omp_set_num_threads(numThreads);
	//__m512d p0, p1, minVec;
	//__m512i clusterIndexVec;

#pragma omp parallel private(i, j, x, y, d, min_d) shared(num_pts, pts, pts_group, numThreads, centroids) firstprivate(clusterIndex) reduction(+ \
																																			   : changes)
	{
#pragma omp for schedule(static, num_pts / numThreads)
		for (i = 0; i < num_pts; i++)
		{

			min_d = 0x7f800000; // IEEE 754 +infinity
			__m512d p0 = _mm512_set1_pd(pts[0][i]);
			__m512d p1 = _mm512_set1_pd(pts[1][i]);
			__m512d minVec = _mm512_set1_pd(min_d);
			__m512i clusterIndexVec = (__m512i)_mm512_set1_epi64(0);

			for (j = 0; j < num_clusters; j += 8)
			{
				__m512d c0 = _mm512_load_pd((centroids[0] + j));
				x = _mm512_sub_pd(c0, p0);
				__m512d c1 = _mm512_load_pd(centroids[1] + j);
				y = _mm512_sub_pd(c1, p1);
				__m512d x2 = _mm512_mul_pd(x, x);
				__m512d y2 = _mm512_mul_pd(y, y);
				__m512d d = _mm512_add_pd(x2, y2);
				__mmask8 mask = _mm512_cmp_pd_mask(d, minVec, 0x1); // mask to make the less than compare
				minVec = _mm512_mask_blend_pd(mask, minVec, d);		// blend to add the result to minVec
				__m512i indices = _mm512_set_epi64(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j);
				clusterIndexVec = _mm512_mask_blend_epi64(mask, clusterIndexVec, indices); // i use the mask again to fix the indices
			}
			int64_t in = getMinAndIndex(minVec, clusterIndexVec);
			if (in != pts_group[i])
			{
				pts_group[i] = in;
				changes++;
			}
		}
	}
	return changes;
}

/*------------------------------------------------------------------
	lloyd findNearestCentroid Dynamic
	This function gets as a parameter the number of threads that the parraler region will create(and creates it with the omp_set_num).
	I decided to parallelize the first for loop(the outer) because every iteration does not have any dependency to any previous iteration. I decided not to
	parallelize the inner for loop because it would have dependency problem with d and min_d  and it would produce wrong results, and if i corrected it it would make
	the program really slow.I decided to make the variables i,j,x,y,d,min_d private as every thread should see its own instance of that variable(eg every thread must see its own
	i and j to produce correct results). I made shared the variables num_pts,pts,pts_group, numThreads,centroids as every thread can see it and access it and all the
	threads should have the same, and the pts_group in which the thread writed data there is no problem because it writes in different index of the array.
	Also the clusterIndex is first private so it can have the value that has before it entered the paraller part and changes is reduction so it can be
	summed every private insant of changes after the paraller region and produce correct results(i decided to use reduction as its an easy , correct and optimized
	way from the omp library).
	This paraller function has dynamic schedule with chunkSize equals to num of points/ num of threads, i added the same chunksize as the static method
	because i think it would be better for the comparison.
	I also added some vectorization in the inner loop, the loop that executes every one thread.
	Before the for loop for the j i make the initializations for the vectors as such p0,p1 for the array pts, i create a minVec which will have in all posiitons
	the min_d and in every loop will have the minimum number of the iterations made so far and the clusterIndexVec is the index for the elements in minVec.
	I do all the calulations as it was before but now in vector. After finding d i have to make the calculations to compare that vector with the vector minVec,
	with the masking as we learnt in the course and i also use that mask to add the data in vector  clusterIndexVec.
	After the while the vector minVec will have the smallest values and clusterIndexVec will have the respectively indeces.
	The i call the function getMinAndIndex to dinf the index of the smallest value, i make the if with ptsgroup[i] and if they are different i add one to the changes
	and put the value of that index in the ptsgroup[i].
	The function getMinAndIndex is explained later in my report.
---------------------------------------------------------------------*/
int FindNearestCentroidSIMD_dynamic(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int numThreads)
{
	int i, j, clusterIndex = 0;
	int changes = 0;
	// double x, y, d, min_d;
	double min_d;
	__m512d x, y, d;
	omp_set_num_threads(numThreads);
	//__m512d p0, p1, minVec;
	//__m512i clusterIndexVec;

#pragma omp parallel private(i, j, x, y, d, min_d) shared(num_pts, pts, pts_group, numThreads, centroids) firstprivate(clusterIndex) reduction(+ \
																																			   : changes)
	{
#pragma omp for schedule(dynamic, num_pts / numThreads)
		for (i = 0; i < num_pts; i++)
		{

			min_d = 0x7f800000; // IEEE 754 +infinity
			__m512d p0 = _mm512_set1_pd(pts[0][i]);
			__m512d p1 = _mm512_set1_pd(pts[1][i]);
			__m512d minVec = _mm512_set1_pd(min_d);
			__m512i clusterIndexVec = (__m512i)_mm512_set1_epi64(0);

			for (j = 0; j < num_clusters; j += 8)
			{
				__m512d c0 = _mm512_load_pd((centroids[0] + j));
				x = _mm512_sub_pd(c0, p0);
				__m512d c1 = _mm512_load_pd(centroids[1] + j);
				y = _mm512_sub_pd(c1, p1);
				__m512d x2 = _mm512_mul_pd(x, x);
				__m512d y2 = _mm512_mul_pd(y, y);
				__m512d d = _mm512_add_pd(x2, y2);
				__mmask8 mask = _mm512_cmp_pd_mask(d, minVec, 0x1); // mask to make the less than compare
				minVec = _mm512_mask_blend_pd(mask, minVec, d);		// blend to add the result to minVec
				__m512i indices = _mm512_set_epi64(j + 7, j + 6, j + 5, j + 4, j + 3, j + 2, j + 1, j);
				clusterIndexVec = _mm512_mask_blend_epi64(mask, clusterIndexVec, indices); // i use the mask again to fix the indices
			}
			int64_t in = getMinAndIndex(minVec, clusterIndexVec);
			if (in != pts_group[i])
			{
				pts_group[i] = in;
				changes++;
			}
		}
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
	lloyd static SIMD
	This is the "parralelized" function-version of lloyd, which gets as a parameter the number of threads and call the function FindNearestCentroidSIMD_static
	which is parralelized statically.
	This function clusters the data using Lloyd's K-Means algorithm
	after selecting the intial centroids.
	It returns a pointer to the memory it allocates containing
	the array of cluster centroids.
---------------------------------------------------------------------*/
void lloyd_ompSIMD_static(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int numThreads)
{
	int i;
	int changes;
	int acceptable = num_pts / 1000; /* The maximum point changes acceptable. */

	if (num_clusters == 1 || num_pts <= 0 || num_clusters > num_pts)
		return;

	if (maxTimes < 1)
		maxTimes = 1;
	/*Assign initial clustering randomly using the Random Partition method*/
	AssignInitialClusteringRandomly_OMP(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, numThreads);

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
		changes = FindNearestCentroidSIMD_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, numThreads);
		maxTimes--;
	} while ((changes > acceptable) && (maxTimes > 0));

	/* Set each centroid's group index */
	for (i = 0; i < num_clusters; i++)
		centroids_group[i] = i;
}
/*------------------------------------------------------------------
	lloyd dynamic SIMD
	This is the "parralelized" function-version of lloyd, which gets as a parameter the number of threads and call the function FindNearestCentroidSIMD_dynamic
	which is parralelized dynamically.
	This function clusters the data using Lloyd's K-Means algorithm
	after selecting the intial centroids.
	It returns a pointer to the memory it allocates containing
	the array of cluster centroids.
---------------------------------------------------------------------*/
void lloyd_pThreadsSIMD_dynamic(double **pts, unsigned int *pts_group, int num_pts, double **centroids, unsigned int *centroids_group, int num_clusters, int maxTimes, int numThreads)
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
		changes = FindNearestCentroidSIMD_dynamic(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, numThreads);

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
	pts_group = malloc(num_pts * sizeof(unsigned int));
	centroids = malloc(nrows * sizeof(double *));
	for (i = 0; i < nrows; i++)
		centroids[i] = aligned_alloc(64, num_clusters * sizeof(double));
	centroids_group = malloc(num_clusters * sizeof(unsigned int));
	startTime(0);
	/* Cluster using the Lloyd algorithm and K-Means++ initial centroids. */

	// lloyd(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes);

	lloyd_ompSIMD_static(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, numThreads);

	//  lloyd_pThreadsSIMD_dynamic(pts, pts_group, num_pts, centroids, centroids_group, num_clusters, maxTimes, numThreads);

	stopTime(0);
	/* Print the results */
	// print_eps_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters);

	print_centroids_v3(pts, pts_group, num_pts, centroids, centroids_group, num_clusters);
	elapsedTime(0);
	free(pts);
	free(centroids);

	//__m512d m1 = _mm512_set_pd(2.0, 10.0, 3.0, 4.0, 1.0, 6.0, 8.0, 7.0);
	//__m512i in = _mm512_set_epi64(0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7);
	// printf("nato %d\n", getMinAndIndex(m1, in));
	// printf("\n");
	return 0;
}
