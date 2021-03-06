#include "stopwatch.h"

#include <values.h>

void StartTimer(StopWatch *timer) {
    struct timezone tz;
    gettimeofday(&(timer->start), &tz);
}

void StopTimer(StopWatch *timer) {
    struct timezone tz;
    gettimeofday(&(timer->stop), &tz);
}

/* Subtrai dois valores de tempo
 * (http://www.gnu.org/software/libc/manual/html_node/Elapsed-Time.html)
 **/

int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y) {
    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }

    /* Compute the time remaining to wait.
    tv_usec is certainly positive. */
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;

    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}

// Retorna o tempo gasto em segundos
double GetElapsedTime(StopWatch *timer) {
    struct timeval result;
    double secs;

    timeval_subtract(&result, &(timer->stop), &(timer->start));

    secs = (double) result.tv_sec;
    secs += ((double) result.tv_usec / (double) 1000000);

    return secs;
}
