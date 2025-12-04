#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include <stdbool.h>

#include "colormap.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <mpi.h>

static bool debug = true;

// Simulation parameters
static const unsigned int N = 500;

static const float SOURCE_TEMP = 5000.0f;
static const float BOUNDARY_TEMP = 1000.0f;

static const float MIN_DELTA = 0.05f;
static const unsigned int MAX_ITERATIONS = 20000;


static unsigned int idx(unsigned int x, unsigned int y, unsigned int stride) {
    return y * stride + x;
}


static void init(unsigned int source_x, unsigned int source_y, float * matrix) {
	// init
	memset(matrix, 0, N * N * sizeof(float));

	// place source
	matrix[idx(source_x, source_y, N)] = SOURCE_TEMP;

	// fill borders
	for (unsigned int x = 0; x < N; ++x) {
		matrix[idx(x, 0,   N)] = BOUNDARY_TEMP;
		matrix[idx(x, N-1, N)] = BOUNDARY_TEMP;
	}
	for (unsigned int y = 0; y < N; ++y) {
		matrix[idx(0,   y, N)] = BOUNDARY_TEMP;
		matrix[idx(N-1, y, N)] = BOUNDARY_TEMP;
	}
}


static void sliced_init(unsigned int source_x, unsigned int source_y, float* matrix, unsigned int slice, unsigned int slice_ind) {
	// init
	unsigned int upper_bound = slice * slice_ind;
	unsigned int lower_bound = slice * (slice_ind + 1);
	if (lower_bound > N) {
		lower_bound = N;
	}

	unsigned int difference = lower_bound - upper_bound;
	memset(matrix, 0, difference * N * sizeof(float));

	// place source
	if (source_y >= upper_bound && source_y < lower_bound)
		matrix[idx(source_x, source_y-upper_bound , N)] = SOURCE_TEMP;

	// fill borders
    if (slice_ind == 0) {
        for (unsigned int x = 0; x < N; ++x) {
            matrix[idx(x, 0, N)] = BOUNDARY_TEMP;
        }
    }
    if (lower_bound == N) {
        for (unsigned int x = 0; x < N; ++x) {
            matrix[idx(x, difference - 1, N)] = BOUNDARY_TEMP;
        }
    }
	for (unsigned int y = 0; y < difference; ++y) {
		matrix[idx(0, y, N)] = BOUNDARY_TEMP;
		matrix[idx(N - 1, y, N)] = BOUNDARY_TEMP;
	}
}


static void step(unsigned int source_x, unsigned int source_y, const float * current, float * next) {

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

/*
static void group_sliced_steps(unsigned int source_x, unsigned int source_y, const float * current, float * next, unsigned int slice, unsigned int slice_ind) {

	float *top_row = (float*)malloc(2*N*sizeof(float));
	float *bottom_row = (float*)malloc(2*N*sizeof(float));

	//float *buffer_recv = (float*)malloc(2*N*sizeof(float));
	
	unsigned int upper_bound = slice * slice_ind;
	unsigned int lower_bound = slice * (slice_ind + 1);
	if (lower_bound > N) {
		lower_bound = N;
	}

	unsigned int pre_ind;
	unsigned int nxt_ind;
	if (slice_ind == 0) {
		pre_ind = (N + slice - 1) / slice - 1;
	}else{
		pre_ind = slice_ind - 1;
	}

	if (slice_ind == (N + slice - 1) / slice - 1) {
		nxt_ind = 0;
	}else{
		nxt_ind = slice_ind + 1;
	}

	for (int ind = 0; ind < N; ind++){
		top_row[ind] = current[idx(ind, slice_ind, N)];
		bottom_row[ind] = current[idx(ind, upper_bound, N)];
	}
	
	MPI_Isend(top_row, N, MPI_FLOAT, pre_ind, 0, MPI_COMM_WORLD);  // tag 0 for the top row
	MPI_Isend(bottom_row, N, MPI_FLOAT, nxt_ind, 1, MPI_COMM_WORLD);  // tag 1 for the bottom row
	
	MPI_Recv(top_row, N, MPI_FLOAT, pre_ind, 0, MPI_COMM_WORLD);  // tag 0 for the top row
	MPI_Recv(bottom_row, N, MPI_FLOAT, nxt_ind, 1, MPI_COMM_WORLD);  // tag 1 for the bottom row

	size_t aux_array_size = (slice + 2) * N * sizeof(float);  // create the new local matrix with ghost rows
	float * aux_matrix = malloc(aux_array_size);
	unsigned int difference = lower_bound - upper_bound;
	for (unsigned int i = 0; i < N; i++) {
		aux_matrix[idx(i, 0, N)] = top_row[i];
		aux_matrix[idx(i, slice + 1, N)] = bottom_row[i];
		for (unsigned int j = 0; j < difference; j++) {
			aux_matrix[idx(i, j + 1, N)] = current[idx(i, upper_bound + j, N)];
		}
	}

	for (unsigned int y = 0; y < slice; ++y) {
		for (unsigned int x = 1; x < N-1; ++x) {
			if ((y == source_y) && (x == source_x)) {
				continue;
			}
			next[idx(x, slice * slice_ind + y, N)] = (aux_matrix[idx(x, y-1, N)] +
			aux_matrix[idx(x-1, y, N)] +
			aux_matrix[idx(x+1, y, N)] +
			aux_matrix[idx(x, y+1, N)]) / 4.0f;
		}
	}

	free(top_row);
    free(bottom_row);
    free(aux_matrix);

}
*/

static void sliced_steps(unsigned int source_x, unsigned int source_y, const float* current, float* next, unsigned int slice, unsigned int slice_ind, unsigned int total_size) {

	float* top_row_send = (float*)malloc(N * sizeof(float));
	float* bottom_row_send = (float*)malloc(N * sizeof(float));
	float* top_row = (float*)malloc(N * sizeof(float));
	float* bottom_row = (float*)malloc(N * sizeof(float));

	MPI_Request reqs[4];

	//float *buffer_recv = (float*)malloc(2*N*sizeof(float));

	unsigned int upper_bound = slice * slice_ind;
	unsigned int lower_bound = slice * (slice_ind + 1);
	if (lower_bound > N) {
		lower_bound = N;
	}

	unsigned int difference = lower_bound - upper_bound;

	unsigned int pre_ind;
	unsigned int nxt_ind;
	/*
	if (slice_ind == 0) {
		pre_ind = (N + slice - 1) / slice - 1;
	}
	else {
		pre_ind = slice_ind - 1;
	}

	if (slice_ind == (N + slice - 1) / slice - 1) {
		nxt_ind = 0;
	}
	else {
		nxt_ind = slice_ind + 1;
	}
    */
	if (slice_ind == 0) {
		pre_ind = total_size - 1;
	}
	else {
		pre_ind = slice_ind - 1;
	}

	if (slice_ind == total_size-1) {
		nxt_ind = 0;
	}
	else {
		nxt_ind = slice_ind + 1;
	}
	for (int ind = 0; ind < N; ind++) {
		top_row_send[ind] = current[idx(ind, 0, N)];
		bottom_row_send[ind] = current[idx(ind, difference-1, N)];
	}

	if(debug) printf("no error at ghost rows creation");

	MPI_Isend(top_row_send, N, MPI_FLOAT, pre_ind, 0, MPI_COMM_WORLD, &reqs[0]);  // tag 0 for the top row
	MPI_Isend(bottom_row_send, N, MPI_FLOAT, nxt_ind, 1, MPI_COMM_WORLD, &reqs[1]);  // tag 1 for the bottom row

	MPI_Irecv(top_row, N, MPI_FLOAT, pre_ind, 0, MPI_COMM_WORLD, &reqs[2]);  // tag 0 for the top row
	MPI_Irecv(bottom_row, N, MPI_FLOAT, nxt_ind, 1, MPI_COMM_WORLD, &reqs[3]);  // tag 1 for the bottom row

	MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

	if(debug) printf("no error at MPI comunication");

	size_t aux_array_size = (difference + 2) * N * sizeof(float);  // create the new local matrix with ghost rows
	float* aux_matrix = malloc(aux_array_size);

	for (unsigned int i = 0; i < N; i++) {
		aux_matrix[idx(i, 0, N)] = top_row[i];
		aux_matrix[idx(i, difference + 1, N)] = bottom_row[i];
		for (unsigned int j = 0; j < difference+1; j++) {
			aux_matrix[idx(i, j + 1, N)] = current[idx(i, j, N)];
		}
	}
	if(debug) printf("no error at aux matrix creation");

	for (unsigned int y = 1; y < difference+1; ++y) {
		for (unsigned int x = 1; x < N - 1; ++x) {
			if ((y+slice*slice_ind-1 == source_y) && (x == source_x)) {
				continue;
			}
			next[idx(x, y, N)] = (aux_matrix[idx(x, y - 1, N)] +
				aux_matrix[idx(x - 1, y, N)] +
				aux_matrix[idx(x + 1, y, N)] +
				aux_matrix[idx(x, y + 1, N)]) / 4.0f;
		}
	}

	free(top_row);
	free(bottom_row);
	free(aux_matrix);

}



static float diff(const float * current, const float * next) {
	float maxdiff = 0.0f;
	for (unsigned int y = 1; y < N-1; ++y) {
		for (unsigned int x = 1; x < N-1; ++x) {
			maxdiff = fmaxf(maxdiff, fabsf(next[idx(x, y, N)] - current[idx(x, y, N)]));
		}
	}
	return maxdiff;
}


static float sliced_diff(const float * current, const float * next, unsigned int slice, unsigned int slice_ind) {

	float maxdiff = 0.0f;

	unsigned int upper_bound = slice * slice_ind;
	unsigned int lower_bound = slice * (slice_ind + 1);
	if (lower_bound > N) {
		lower_bound = N;
	}

	for (unsigned int y = upper_bound + 1; y < lower_bound - 1; ++y) {
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


int main(int argc, char **argv) {
	size_t array_size = N * N * sizeof(float);

	//float * current = malloc(array_size);
	//float * next = malloc(array_size);

	float* global_current = malloc(array_size);

	srand(0);
	unsigned int source_x = rand() % (N-2) + 1;
	unsigned int source_y = rand() % (N-2) + 1;
	printf("Heat source at (%u, %u)\n", source_x, source_y);

	float t_diff = SOURCE_TEMP;

	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int num_per_slice;
	if (size == 1) {
		num_per_slice = N;
	}else {
		num_per_slice = N / size + 1;
	}

	size_t slice_size;

	if (num_per_slice * (rank + 1) < N) {
		slice_size = num_per_slice * N * sizeof(float);
	}
	else {
		slice_size = (N-num_per_slice*rank) * N * sizeof(float);
	}

	float* current = malloc(slice_size);
	float* next = malloc(slice_size);

	sliced_init(source_x, source_y, current, num_per_slice, rank);

	if(debug) printf("no error init \n");

	memcpy(next, current, slice_size);

	double start = omp_get_wtime();

	float sliced_t_diff = SOURCE_TEMP;

	for (unsigned int it = 0; (it < MAX_ITERATIONS) && (t_diff > MIN_DELTA); ++it) {
		/*
		step(source_x, source_y, current, next);

		t_diff = diff(current, next);
		if(it%(MAX_ITERATIONS/10)==0){
			printf("%u: %f\n", it, t_diff);
		}
		*/

		/********below*********/
		sliced_steps(source_x, source_y, current, next, num_per_slice, rank, size);

		printf("no error step");
		sliced_t_diff = sliced_diff(current, next, num_per_slice, rank);
		printf("no error diff");
		MPI_Reduce(&sliced_t_diff, &t_diff, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
		if (it % (MAX_ITERATIONS / 10) == 0) {
			printf("%u: %f\n", it, t_diff);
		}
		/********above*********/

		float * swap = current;
		current = next;
		next = swap;

		if(debug) printf("no error at %d iter of sliced_step\n", it);

	}
	double stop = omp_get_wtime();
	printf("Computing time %f s.\n", stop-start);

	int* recvcounts = NULL;
	int* displs = NULL;

	if (rank == 0) {
		recvcounts = malloc(size * sizeof(int));
		displs = malloc(size * sizeof(int));

		int offset = 0;

		for (int i = 0; i < size; i++) {
			if (num_per_slice * (i + 1) < N) {
				recvcounts[i] = num_per_slice * N;
			}
			else {
				recvcounts[i] = (N - num_per_slice * i) * N;
			}
			
			displs[i] = offset;
			offset += recvcounts[i];
		}
	}

	MPI_Gatherv(current, slice_size / sizeof(float), MPI_FLOAT, global_current, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		write_png(global_current, MAX_ITERATIONS);
	}

	free(current);
	free(next);
	MPI_Finalize();

	free(global_current);

	return 0;
}
