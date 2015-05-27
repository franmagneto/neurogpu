/*

  neurogpu++.cu
  Interface for the neurogpu++ program, that implements MLP Neural Networks
  in CUDA.

  Francisco M. Magalhaes Neto, 2014-05-28
  Based on neurogpu, by Andrei de A. Formiga, 2012-05-21

 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlpnnets.h"
#include "stopwatch.h"

#define SEED                  631814ULL
#define MAX_ABS               1.2f

DataSet* readDataset(char *filename, int inputSize, int outputSize, bool isFunction)
{
	FILE    *f;
	int     i;
	char    buffer[500];
	char    *value;
	float   outputValue;
	float   *outputs;
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

	int  iix = 0, oix = 0;
	outputs = (float*) malloc(sizeof(float) * outputSize);
	while(fgets(buffer, 500, f) != NULL){

		value = strtok(buffer, ",");
		for (int i = 0; i < inputSize; ++i) {

			dset->inputs[iix++] = atof(value);
			value = strtok(NULL, ",");
		}
		outputValue = atof(value);
		if (outputSize > 1)
			for (int i = 0; i < outputSize; ++i)
				dset->outputs[oix++] = (i == outputValue) ? 0.9f : 0.1f;
		else if (!isFunction)
			dset->outputs[oix++] = (outputValue == 1) ? 0.9f : 0.1f;
		else
			dset->outputs[oix++] = (outputValue + 1.0f) / 2.0f;
	}

	free(outputs);
	fclose(f);

	return dset;
}

void print_dataset(DataSet *dset)
{
	int i, j;

	printf("Number of cases: %d\n", dset->nCases);
	for (i = 0; i < dset->nCases; ++i) {
		for (j = 0; j < dset->inputSize; ++j)
			printf("%3.2f ", dset->inputs[i*dset->inputSize+j]);
		printf(" | ");
		for (j = 0; j < dset->outputSize; ++j)
			printf("%3.2f ", dset->outputs[i*dset->outputSize+j]);
		printf("\n");
	}
}

int outputToClass(float *output, int outputSize)
{
	int classNumber = 0;

	if (outputSize == 1) {
		if (output[0] < 0.5f)
			return 0;

		return 1;
	}

	for (int i = 0; i < outputSize; ++i) {
		if (output[i] > output[classNumber])
			classNumber = i;
	}
	return classNumber;
}

void print_network_data(MLPNetwork *net)
{
	printf("nLayers = %d, d_weights = %lu, nWeights = %d, nCases = %d\n",
	       net->nLayers, (unsigned long) net->d_weights, net->nWeights, net->nCases);
	printf("output ptr for first layer: %lu\n", (unsigned long) net->layers[0]->d_outs);
	printf("output ptr for last layer: %lu\n", (unsigned long) net->layers[net->nLayers-1]->d_outs);
}

struct TestDescription {
	int inputSize;
	int outputSize;
	int epochs;
	float learningRate;
	bool isFunction;
	int nLayers;
	int *neuronsPerLayer;
};

TestDescription *readDescription(const char *name) {
	FILE *f;
	TestDescription *desc;
	char buf[50];
	char *layer;
	int isFunction;

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
	for (int i = 0; i < desc->nLayers; ++i) {
		desc->neuronsPerLayer[i] = atoi(layer);
		layer = strtok(NULL, ",");
	}

	return desc;
}

void destroyDescription(TestDescription *desc) {
	free(desc->neuronsPerLayer);
	free(desc);
}

int runTest(const char *name, int casesPerBlock, int neuronsPerThread) {
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

	MLPNetwork *nn;

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
	RandomWeights(nn, MAX_ABS, SEED);

	printf("Training network with %d epochs...\n", desc->epochs);
	StartTimer(&timer);
	e = BatchTrainBackprop(nn, train_set, desc->epochs, desc->learningRate,
			true, false, ACTF_SIGMOID, casesPerBlock, neuronsPerThread);
	StopTimer(&timer);
	elapsedTime = GetElapsedTime(&timer);
	printf("Training finished, approximate final MSE: %f\n", e/nn->nCases);

	printf("Weights after training:\n");
	PrintWeights(nn);

	printf("-----------------------------------------\n");

	// free the training dataset
	cudaThreadSynchronize();
	DestroyDataSet(train_set);

	// testing
	test_set = readDataset(test_name, desc->inputSize, desc->outputSize, desc->isFunction);

	if (test_set == NULL) {
		fprintf(stderr, "Error reading test set\n");
		return -1;
	}

	errors = 0;

	if (!PrepareForTesting(nn, test_set->nCases)) {
		fprintf(stderr, "Error preparing network for testing\n");
		return -1;
	}

	printf("Testing with %d cases...\n", test_set->nCases);
	PresentInputsFromDataSet(nn, test_set, ACTF_SIGMOID, 1, 1);

	cudaThreadSynchronize();

	printf("Weights again:\n");
	PrintWeights(nn);

	float *output = (float*) malloc(sizeof(float) * test_set->nCases * test_set->outputSize);

	if (output == NULL) {
		fprintf(stderr, "Could not allocate memory for copying output to host\n");
		return -1;
	}

	if (!CopyNetworkOutputs(nn, output)) {
		fprintf(stderr, "Could not get device outputs\n");
		return -1;
	}

	if (!desc->isFunction) {
		int predicted, desired;
		for (i = 0; i < test_set->nCases; ++i) {
			predicted = outputToClass(output + (i * desc->outputSize), desc->outputSize);
			desired = outputToClass(test_set->outputs + (i * desc->outputSize), desc->outputSize);
			if (predicted != desired)
				++errors;
			printf("Case %d | predicted: %d, desired: %d\n", i,
				   predicted, desired);
		}

		acc = 100.0 - (100.0 * errors / test_set->nCases);
		printf("Testing accuracy: %f\n", acc);
		printf("Total classification errors: %d\n", errors);
	} else {
		float predicted, desired;
		double error = 0.0f;
		for (i = 0; i < test_set->nCases; ++i) {
			predicted = output[i];
			desired = test_set->outputs[i];
			error += fabs(predicted - desired);
			printf("Input %f | predicted: %f, desired: %f\n", test_set->inputs[i],
			       predicted, desired);
		}
		error /= test_set->nCases;
		printf("Mean error: %f\n", error);
	}

	free(output);
	DestroyNetwork(nn);
	DestroyDataSet(test_set);
	destroyDescription(desc);

	fprintf(stderr, "Training time: %lf seconds\n", elapsedTime);

	return 0;
}

int main(int argc, char **argv)
{
	int casesPerBlock = 1;
	int neuronsPerThread = 1;
	char name[FILENAME_MAX] = "cancer";

	if (argc > 1) {
		strcpy(name, argv[1]);
		if (argc == 4) {
			casesPerBlock = atoi(argv[2]);
			neuronsPerThread = atoi(argv[3]);
		} else if (argc != 2) {
			fprintf(stderr, "Syntax error\n");
			exit(EXIT_FAILURE);
		}
	}

	if ((casesPerBlock > 0) && (neuronsPerThread > 0))
		return runTest(name, casesPerBlock, neuronsPerThread);
	fprintf(stderr, "Syntax error\n");
	return EXIT_FAILURE;
}
