#include <stdint.h>

void render_board(
    uint16_t *board, uint16_t *goals,
    int width, int height, int depth,
    float *sprites, uint8_t *out);

extern const int SPRITE_SIZE;
