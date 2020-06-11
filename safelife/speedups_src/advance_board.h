#include <stdint.h>

void advance_board(
    uint16_t *b1, uint16_t *b2, int height, int width, float spawn_prob);

void alive_counts(uint16_t *board, uint16_t *goals, int n, int64_t *out);
