#include <stdint.h>

#define MAX_ITER_ERROR -1
#define PROBABILITY_ERROR -2
#define AREA_TOO_SMALL_ERROR -3

int gen_still_life(
        int16_t *board, int32_t *mask, int nrow, int ncol,
        double rel_max_iter, double rel_min_fill, double temperature,
        double *cell_penalties);
