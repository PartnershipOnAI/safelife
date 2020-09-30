#include <stdint.h>

void advance_board(
    uint16_t *b1, uint16_t *b2, int height, int width, float spawn_prob, uint16_t *c0);

void advance_board_nstep(
    uint16_t *b1, uint16_t *b2, int height, int width, float spawn_prob, int n_steps);

void life_occupancy(
        uint16_t *b1, int32_t *counts, int nrow, int ncol, float spawn_prob, int n_steps);

void alive_counts(uint16_t *board, uint16_t *goals, int n, int64_t *out);

void execute_actions(
    uint16_t *board, int w, int h,
    int64_t *locations, int64_t *actions, int n_agents, int action_stride);
