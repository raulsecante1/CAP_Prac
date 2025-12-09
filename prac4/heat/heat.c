#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "colormap.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Simulation parameters
static const unsigned int N = 500;

static const float SOURCE_TEMP = 5000.0f;
static const float BOUNDARY_TEMP = 1000.0f;

static const float MIN_DELTA = 0.05f;
static const unsigned int MAX_ITERATIONS = 20000;


#pragma omp declare target
static void step(unsigned int source_x, unsigned int source_y, const float * current, float * next);
static float diff(const float * current, const float * next);
#pragma omp end declare target


static unsigned int idx(unsigned int x, unsigned int y, unsigned int stride) {
    return y * stride + x;
}


static void init(unsigned int source_x, unsigned int source_y, float * matrix) {
	// init
	memset(matrix, 0, N * N * sizeof(float));

	// place source
	matrix[idx(source_x, source_y, N)] = SOURCE_TEMP;

	// fill borders
	#pragma omp target teams distribute parallel for map(tofrom: matrix[0:N*N])
	for (unsigned int x = 0; x < N; ++x) {
		matrix[idx(x, 0,   N)] = BOUNDARY_TEMP;
		matrix[idx(x, N-1, N)] = BOUNDARY_TEMP;
	}
	#pragma omp target teams distribute parallel for map(tofrom: matrix[0:N*N])
	for (unsigned int y = 0; y < N; ++y) {
		matrix[idx(0,   y, N)] = BOUNDARY_TEMP;
		matrix[idx(N-1, y, N)] = BOUNDARY_TEMP;
	}
}


static void step(unsigned int source_x, unsigned int source_y, const float * current, float * next) {

	//#pragma omp target teams distribute parallel for collapse(2) map(to: current[0:N*N]) map(tofrom: next[0:N*N])
	for (unsigned int y = 1; y < N-1; ++y) {
		for (unsigned int x = 1; x < N-1; ++x) {
			if ((y == source_y) && (x == source_x)) {
				continue;
			}
			next[idx(x, y, N)] = (current[idx(x, y-1, N)] +
			current[idx(x-1, y, N)] +
			current[idx(x+1, y, N)] +
			current[idx(x, y+1, N)]) / 4.0f;
		}
	}
}


static float diff(const float * current, const float * next) {
	float maxdiff = 0.0f;

    //#pragma omp target teams distribute parallel for collapse(2)reduction(max:maxdiff) map(to: next[0:N*N], current[0:N*N])
	for (unsigned int y = 1; y < N-1; ++y) {
		for (unsigned int x = 1; x < N-1; ++x) {
			maxdiff = fmaxf(maxdiff, fabsf(next[idx(x, y, N)] - current[idx(x, y, N)]));
		}
	}
	return maxdiff;
}


void write_png(float * current, int iter) {
	char file[100];
	uint8_t * image = malloc(3 * N * N * sizeof(uint8_t));
	float maxval = fmaxf(SOURCE_TEMP, BOUNDARY_TEMP);

	for (unsigned int y = 0; y < N; ++y) {
		for (unsigned int x = 0; x < N; ++x) {
			unsigned int i = idx(x, y, N);
			colormap_rgb(COLORMAP_MAGMA, current[i], 0.0f, maxval, &image[3*i], &image[3*i + 1], &image[3*i + 2]);
		}
	}
	sprintf(file,"heat%i.png", iter);
	stbi_write_png(file, N, N, 3, image, 3 * N);

	free(image);
}


int main() {
	size_t array_size = N * N * sizeof(float);

	float * current = malloc(array_size);
	float * next = malloc(array_size);

	srand(0);
	unsigned int source_x = rand() % (N-2) + 1;
	unsigned int source_y = rand() % (N-2) + 1;
	printf("Heat source at (%u, %u)\n", source_x, source_y);

	init(source_x, source_y, current);
	memcpy(next, current, array_size);

	double start = omp_get_wtime();

	float t_diff = SOURCE_TEMP;

	#pragma omp target data map(to: current[0:N*N]) map(alloc: next[0:N*N])
	{
	for (unsigned int it = 0; (it < MAX_ITERATIONS) && (t_diff > MIN_DELTA); ++it) {
		step(source_x, source_y, current, next);
		t_diff = diff(current, next);
		if(it%(MAX_ITERATIONS/10)==0){
			printf("%u: %f\n", it, t_diff);
		}

		float * swap = current;
		current = next;
		next = swap;
	}
    }
	double stop = omp_get_wtime();
	printf("Computing time %f s.\n", stop-start);

	write_png(current, MAX_ITERATIONS);

	free(current);
	free(next);

	return 0;
}
