/*
 * stopwatch.h
 *
 *  Created on: 29/01/2015
 *      Author: francisco
 */

#ifndef STOPWATCH_H_
#define STOPWATCH_H_

#include <sys/time.h>

typedef struct {
    struct timeval start;
    struct timeval stop;
} StopWatch;

void StartTimer(StopWatch*);

void StopTimer(StopWatch*);

double GetElapsedTime(StopWatch*);

#endif /* STOPWATCH_H_ */
