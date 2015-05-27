CUDA	= nvcc -c -O3 -arch sm_13
CC		= g++ -c -O3

all: neurogpu-plus

neurogpu-plus: neurogpu-plus.o mlpnnets.o stopwatch.o
	nvcc -o neurogpu-plus neurogpu-plus.o mlpnnets.o stopwatch.o -lcurand

neuro: neuro.o mlpnnets_cpu.o stopwatch.o
	g++ -o neuro neuro.o mlpnnets_cpu.o stopwatch.o

neurogpu-plus.o: neurogpu-plus.cu mlpnnets.h
	$(CUDA) neurogpu-plus.cu

neuro.o: neuro.c mlpnnets_cpu.h
	$(CC) neuro.c

mlpnnets.o: mlpnnets.cu mlpnnets.h
	$(CUDA) mlpnnets.cu -lcurand

mlpnnets_cpu.o: mlpnnets_cpu.c mlpnnets_cpu.h
	$(CC) mlpnnets_cpu.c

stopwatch.o: stopwatch.c stopwatch.h
	$(CC) stopwatch.c

sin-gen: sin-gen.cu
	nvcc -o sin-gen sin-gen.cu

clean:
	rm -f *.o
	rm -f neurogpu-plus
	rm -f neuro
	rm -f sin-gen
