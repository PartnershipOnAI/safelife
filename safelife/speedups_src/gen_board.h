#include <stdint.h>

enum gen_board_errors {
    MAX_ITER_ERROR = -1,
    PROBABILITY_ERROR = -2,
    AREA_TOO_SMALL_ERROR = -3,
};

enum gen_mask_bits {
    NEW_CELL_MASK = 1,
    CAN_OSCILLATE_MASK = 2,
    INCLUDE_VIOLATIONS_MASK = 4,
};

typedef struct {
    int depth;
    int rows;
    int cols;
} board_shape_t;

int gen_pattern(
        uint16_t *board, int32_t *mask, int32_t *seeds, board_shape_t shape,
        double rel_max_iter, double rel_min_fill, double temperature, double osc_bonus,
        double *cell_penalties);
