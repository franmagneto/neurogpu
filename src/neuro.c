/*

  neuro++.cu
  Interface for the neuro++ program, that implements MLP Neural Networks
  in C.

  Francisco M. Magalhaes Neto, 2014-05-28
  Based on neurogpu, by Andrei de A. Formiga, 2012-05-21

 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mlpnnets_cpu.h"
#include "stopwatch.h"

#define TRUE    1
#define FALSE   0

#define SEED                  631814ULL
#define MAX_ABS               1.2f

typedef struct tagTestDescription {
	int inputSize;
	int outputSize;
	int epochs;
	float learningRate;
	int isFunction;
	int nLayers;
	int *neuronsPerLayer;
} TestDescription;

// Create a MLP neural network for execution on the CPU.
// n_layers: number of layers
// neuronsPerLayer: array of ints (size equal to n_layers) with the
//                  number of neurons for each layer
Network *CreateNetwork(int n_layers, int *neuronsPerLayer)
{
	int i, n_inputs = neuronsPerLayer[0];
	Network *res = (Network*) malloc(sizeof(Network));

	res->n_layers = 1;
	res->input_layer = (Layer*) malloc(sizeof(Layer));
	res->input_layer->n_neurons = n_inputs;
	res->input_layer->w = NULL;
	res->input_layer->prev = NULL;
	res->input_layer->next = NULL;
	res->input_layer->a = NULL;
	res->input_layer->y = (double*) malloc(sizeof(double) * n_inputs);

	res->output_layer = res->input_layer;

	for (i = 1; i < n_layers; i++) {
		add_layer(res, neuronsPerLayer[i]);
	}

	return res;
}

DataSet* CreateDataSet(int n_cases, int input_size, int output_size)
{
	DataSet *result;

	result = (DataSet*) malloc(sizeof(DataSet));

	if (result == NULL)
		return NULL;

	result->n_cases = n_cases;
	result->input_size = input_size;
	result->output_size = output_size;

	allocate_dataset_arrays(result);

	if (result->input == NULL) {
		free(result);
		return NULL;
	}

	if (result->output == NULL) {
		free(result->input);
		free(result);
		return NULL;
	}

	return result;
}

DataSet* readDataset(char *filename, int inputSize, int outputSize, int isFunction)
{
	FILE    *f;
	int     i;
	char    buffer[500];
	char    *value;
	float   outputValue;
	DataSet *dset;

	f = fopen(filename, "r");
	if (f == NULL) {
		fprintf(stderr, "File not found: %s\n", filename);
		return NULL;
	}

	// count lines in file to allocate dataset arrays
	i = 0;
	while (fgets(buffer, 500, f) != NULL)
		++i;

	if (!feof(f) || ferror(f)) {
		fprintf(stderr, "IO error while reading from file\n");
		fclose(f);
		return NULL;
	}
	fseek(f, 0, SEEK_SET);

	dset = CreateDataSet(i, inputSize, outputSize);

	if (dset == NULL) {
		fprintf(stderr, "Error creating dataset\n");
		return NULL;
	}

	int  ix = 0;
	while(fgets(buffer, 500, f) != NULL) {
		int i;

		value = strtok(buffer, ",");
		for (i = 0; i < inputSize; ++i) {
			
			dset->input[ix][i] = atof(value);
			value = strtok(NULL, ",");
		}
		outputValue = atof(value);
		if (outputSize > 1)
			for (i = 0; i < outputSize; ++i)
				dset->output[ix][i] = (i == outputValue) ? 0.9f : 0.1f;
		else if (!isFunction)
			dset->output[ix][0] = (outputValue == 1) ? 0.9f : 0.1f;
		else
			dset->output[ix][0] = (outputValue + 1.0f) / 2.0f;
		ix++;
	}
	fclose(f);

	return dset;
}

void print_dataset(DataSet *dset)
{
	int i, j;

	printf("Number of cases: %d\n", dset->n_cases);
	for (i = 0; i < dset->n_cases; ++i) {
		for (j = 0; j < dset->input_size; ++j)
			printf("%3.2lf ", dset->input[i][j]);
		printf(" | ");
		for (j = 0; j < dset->output_size; ++j)
			printf("%3.2lf ", dset->output[i][j]);
		printf("\n");
	}
}

int outputToClass(double *output, int outputSize)
{
	int classNumber = 0;
	int i;

	if (outputSize == 1) {
		if (output[0] < 0.5f)
			return 0;

		return 1;
	}

	for (i = 0; i < outputSize; ++i) 
		if (output[i] > output[classNumber])
			classNumber = i;

	return classNumber;
}

void print_network_data(Network *net)
{
	printf("n_layers = %d\n", net->n_layers);
	printf("output ptr for first layer: %lu\n", (unsigned long) net->output_layer);
	printf("output ptr for last layer: %lu\n", (unsigned long) net->input_layer);
}

TestDescription *readDescription(const char *name) {
	FILE *f;
	TestDescription *desc;
	char buf[50];
	char *layer;
	int isFunction;
	int i;

	f = fopen(name, "r");
	if (f == NULL) {
		fprintf(stderr, "File not found: %s\n", name);
		return NULL;
	}
	desc = (TestDescription*) malloc(sizeof(TestDescription));

	fscanf(f, "%d\n", &desc->inputSize);
	fscanf(f, "%d\n", &desc->outputSize);
	fscanf(f, "%d\n", &desc->epochs);
	fscanf(f, "%f\n", &desc->learningRate);

	fscanf(f, "%d\n", &isFunction);
	desc->isFunction = isFunction;

	fscanf(f, "%d\n", &desc->nLayers);

	desc->neuronsPerLayer = (int*) malloc(sizeof(int) * desc->nLayers);

	fgets(buf, 50, f);
	layer = strtok(buf, ",");
	for (i = 0; i < desc->nLayers; ++i) {
		desc->neuronsPerLayer[i] = atoi(layer);
		layer = strtok(NULL, ",");
	}

	return desc;
}

void destroyDescription(TestDescription *desc) {
	free(desc->neuronsPerLayer);
	free(desc);
}

int runTest(const char *name) {
	int     i;
	int     errors;
	DataSet *train_set;
	DataSet *test_set;
	TestDescription *desc;
	float   e;
	double  acc;
	StopWatch timer;
	double elapsedTime;
	char desc_name[FILENAME_MAX];
	char train_name[FILENAME_MAX];
	char test_name[FILENAME_MAX];

	Network *nn;

	sprintf(desc_name, "data/%s.desc", name);
	sprintf(train_name, "data/%s.train", name);
	sprintf(test_name, "data/%s.test", name);

	desc = readDescription(desc_name);

	// training
	train_set = readDataset(train_name, desc->inputSize, desc->outputSize, desc->isFunction);

	if (train_set == NULL) {
		fprintf(stderr, "Error reading training set\n");
		exit(1);
	}

	nn = CreateNetwork(desc->nLayers, desc->neuronsPerLayer);
	initialize_weights(nn, SEED);

	printf("Training network with %d epochs...\n", desc->epochs);
	StartTimer(&timer);
	e = batch_train(nn, train_set, desc->learningRate, desc->epochs,
			sigmoid, dsigmoid);
	StopTimer(&timer);
	elapsedTime = GetElapsedTime(&timer);
	printf("Training finished, approximate final MSE: %f\n", e/train_set->n_cases);
	
	printf("-----------------------------------------\n");

	// free the training dataset
	free_dataset(train_set);

	// testing
	test_set = readDataset(test_name, desc->inputSize, desc->outputSize, desc->isFunction);

	if (test_set == NULL) {
		fprintf(stderr, "Error reading test set\n");
		return -1;
	}

	errors = 0;

	printf("Testing with %d cases...\n", test_set->n_cases);

	if (!desc->isFunction) {
		int predicted, desired;
		for (i = 0; i < test_set->n_cases; ++i) {
			forward_prop(nn, sigmoid, test_set->input[i]);
			desired = outputToClass(test_set->output[i], desc->outputSize);
			predicted = outputToClass(nn->output_layer->y, desc->outputSize);
			if (predicted != desired)
				++errors;
			printf("Case %d | predicted: %d, desired: %d\n", i,
				   predicted, desired);
		}

		acc = 100.0 - (100.0 * errors / test_set->n_cases);
		printf("Testing accuracy: %f\n", acc);
		printf("Total classification errors: %d\n", errors);
	} else {
		float predicted, desired;
		double error = 0.0f;
		for (i = 0; i < test_set->n_cases; ++i) {
			predicted = nn->output_layer->y[0];
			desired = test_set->output[i][0];
			error += fabs(predicted - desired);
			printf("Input %f | predicted: %f, desired: %f\n", test_set->input[i][0],
			       predicted, desired);
		}
		error /= test_set->n_cases;
		printf("Mean error: %f\n", error);
	}

	destroy_network(nn);
	free_dataset(test_set);
	destroyDescription(desc);

	fprintf(stderr, "Training time: %lf seconds\n", elapsedTime);

	return 0;
}

int main(int argc, char **argv)
{
	char name[FILENAME_MAX] = "cancer";

	if (argc > 1) {
		strcpy(name, argv[1]);
		if (argc != 2) {
			fprintf(stderr, "Syntax error\n");
			exit(EXIT_FAILURE);
		}
	}

	return runTest(name);
}
