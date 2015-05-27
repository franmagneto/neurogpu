#include <stdio.h>
#include <stdlib.h>

#define MIN(a, b) (a < b) ? a : b

__global__ void inputs_gen(float *in, int samples, float first, float last) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	float precision = (last - first) / (samples-1);

	if (tid < samples) {
		in[tid] = (tid*precision + first);
	}
}

__global__ void sin_compute(float *in, float *out, int samples) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < samples) {
		out[tid] = sinf(in[tid]);
	}
}

int main(int argc, char **argv) {

	float *in, *out;
	float *d_in, *d_out;
	int samples = 12501;
	char option;
	cudaError_t err;

	if (argc == 2) {
		option = argv[1][0];
		switch (option) {
			case 'a':
				samples = 12500001;
				break;
			case 'b':
				samples = 1250001;
				break;
			case 'c':
				samples = 125001;
				break;
			case 'd':
				samples = 12501;
				break;
			case 'e':
			default:
				samples = 1251;
		}
	}
	int size = sizeof(float)*samples;

	in = (float*)malloc(size);
	out = (float*)malloc(size);

	if (!in || !out) {
		fprintf(stderr, "Erro alocando vetores\n");
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc(&d_in, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Erro alocando entradas (%s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc(&d_out, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Erro alocando saidas (%s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	int tpb = MIN(samples, 512);
    int bpg = (samples-1)/tpb+1;
	inputs_gen<<<bpg, tpb>>>(d_in, samples, -2.0 * M_PI, 2 * M_PI);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Erro ao gerar entradas\n");
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();

	sin_compute<<<bpg, tpb>>>(d_in, d_out, samples);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Erro ao computar seno\n");
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();

	err = cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Erro ao transferir valores de entrada (%s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Erro ao transferir valores de saida (%s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaFree(d_in);
	cudaFree(d_out);

	for (int i = 0; i < samples; ++i) {
		printf("%f,%f\n", in[i], out[i]);
	}
	free(in);
	free(out);

	return EXIT_SUCCESS;
}
