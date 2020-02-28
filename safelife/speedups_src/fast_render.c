#include "fast_render.h"

#define COLOR_BIT 9

const int SPRITE_SIZE = 14;

float foreground_colors[] = {
    0.4, 0.4, 0.4,  // black
    0.8, 0.2, 0.2,  // red
    0.2, 0.8, 0.2,  // green
    0.8, 0.8, 0.2,  // yellow
    0.2, 0.2, 0.8,  // blue
    0.8, 0.2, 0.8,  // magenta
    0.2, 0.8, 0.8,  // cyan
    1.0, 1.0, 1.0,  // white
};

float background_colors[] = {
    0.6, 0.6, 0.6,  // black
    0.9, 0.6, 0.6,  // red
    0.6, 0.9, 0.6,  // green
    0.9, 0.9, 0.6,  // yellow
    0.5, 0.5, 0.9,  // blue
    0.9, 0.6, 0.9,  // magenta
    0.6, 0.9, 0.9,  // cyan
    0.9, 0.9, 0.9,  // white
};

uint16_t color_mask = 7 << COLOR_BIT;

static inline void draw_sprite(
        uint16_t cell, uint16_t goal, uint8_t orientation,
        float *sprites, uint8_t *out, int row_stride) {
    float *fg_color = foreground_colors + ((cell & color_mask) >> COLOR_BIT) * 3;
    float *bg_color = background_colors + ((goal & color_mask) >> COLOR_BIT) * 3;
    cell &= ~color_mask;

    int row, col;
    switch (cell) {
        case 0:  // empty
            row = 0; col = 0; break;

        case 9:  // life
            row = 1; col = 0; break;
        case 1:  // hard life
            row = 1; col = 1; break;
        case 53:  // weed
            row = 1; col = 2; break;
        case 32789:  // plant
            row = 1; col = 3; break;
        case 17:  // tree
            row = 1; col = 4; break;

        case 32884:  // ice cube
            row = 2; col = 0; break;
        case 48:  // fountain
            row = 2; col = 1; break;
        case 16:  // wall
            row = 2; col = 2; break;
        case 32788:  // crate
            row = 2; col = 3; break;
        case 85:  // parasite
            row = 2; col = 4; break;

        case 152:  // spawner
            row = 3; col = 0; break;
        case 272:  // exit
            row = 3; col = 1; break;
        case 144:  // hard spawner
            row = 3; col = 2; break;

        default:
            if (cell & 2) {  // agent
                row = 0; col = 1 + orientation;
            } else {  // unknown
                row = 3; col = 4;
            }
    }

    float *sprite = sprites + (5*row * SPRITE_SIZE + col) * SPRITE_SIZE * 4;
    int sprite_row = SPRITE_SIZE * 4 * 4;
    row_stride -= SPRITE_SIZE * 3;

    for (int r=0; r<SPRITE_SIZE; r++) {
        for (int c=0; c<SPRITE_SIZE; c++) {
            float mask = sprite[3];

            out[0] = 255 * (bg_color[0] * (1-mask) + mask * sprite[0] * fg_color[0]);
            out[1] = 255 * (bg_color[1] * (1-mask) + mask * sprite[1] * fg_color[1]);
            out[2] = 255 * (bg_color[2] * (1-mask) + mask * sprite[2] * fg_color[2]);

            out += 3;
            sprite += 4;
        }
        out += row_stride;
        sprite += sprite_row;
    }
}


void render_board(
        uint16_t *board, uint16_t *goals, uint8_t *orientation,
        int width, int height, int depth,
        float *sprites, uint8_t *out) {

    // Shapes should be:
    //     board = (depth, height, width)
    //     goals = (depth, height, width)
    //     orientation = (depth,)
    //     sprites = (5 * SPRITE_SIZE, 5 * SPRITE_SIZE, 4)
    //     out = (depth, height * SPRITE_SIZE, width * SPRITE_SIZE, 3)

    int out_stride = width * SPRITE_SIZE * 3;

    for (int i=0; i < depth; i++) {
        for (int j=0; j < height; j++) {
            for (int k=0; k < width; k++) {
                draw_sprite(*board, *goals, *orientation, sprites, out, out_stride);
                board++;
                goals++;
                out += SPRITE_SIZE * 3;
            }
            out += out_stride * (SPRITE_SIZE - 1);
        }
        orientation++;
    }
}
