#include <string.h>
#include "advance_board.h"
#include "constants.h"
#include "random.h"

static const uint16_t ALIVE_BITS = (1 << 4) - 1;
static const uint16_t DESTRUCTIBLE2 = 1 << 8;
static const uint16_t FLAGS1 = PRESERVING | INHIBITING | SPAWNING;
static const uint16_t FLAGS2 = (1 << 8) | COLORS;


static void combine_neighbors(uint16_t src, uint16_t *dst) {
    // Combine neighbors with base board.
    uint16_t alive = src & ALIVE;
    uint16_t src_flags1 = src & FLAGS1;
    uint16_t src_flags2 = (src & FLAGS2) * alive;
    uint16_t dst_flags2 = *dst & FLAGS2;
    *dst |= (dst_flags2 & src_flags2) << 4;
    *dst |= ((src & COLORS) << 4) * ((src & SPAWNING) > 0);
    *dst |= src_flags1;
    *dst |= src_flags2;
    *dst += alive;
}


static void combine_neighbors2(uint16_t src, uint16_t *dst) {
    // Combine combinations.
    *dst |= (*dst & src & FLAGS2) << 4;
    *dst |= src & (FLAGS1 | FLAGS2 | (FLAGS2 << 4));
    *dst += src & ALIVE_BITS;

}

void advance_board(
        uint16_t *b1, uint16_t *b2, int nrow, int ncol, float spawn_prob) {
    int size = nrow*ncol;
    int i, j, start_of_row, end_of_row, end_of_col;
    uint16_t c1[size];
    memset(c1, 0, sizeof(c1));

    // Adjust all of the bits in b2 so that the destructible bit overwrites
    // the exit bit. This allows us to treat destructibility and colors at
    // the same time.
    for (i = 0; i < size; i++) {
        b2[i] = b1[i] | (b1[i] & DESTRUCTIBLE) << 5;
    }

    // First figure out what the neighboring bits are.
    // Can do this in two 1-d convolutions.

    // Combine along rows
    for (i = 0; i < nrow; i++) {
        // Wrap at start of row
        start_of_row = i * ncol;
        end_of_row = start_of_row + ncol - 1;
        combine_neighbors(b2[start_of_row], c1+start_of_row);
        combine_neighbors(b2[start_of_row+1], c1+start_of_row);
        combine_neighbors(b2[end_of_row], c1+start_of_row);
        // Loop over middle
        for (j = start_of_row + 1; j < end_of_row; j++) {
            combine_neighbors(b2[j], c1+j);
            combine_neighbors(b2[j+1], c1+j);
            combine_neighbors(b2[j-1], c1+j);
        }
        // Wrap at end of row
        combine_neighbors(b2[end_of_row], c1+end_of_row);
        combine_neighbors(b2[end_of_row-1], c1+end_of_row);
        combine_neighbors(b2[start_of_row], c1+end_of_row);
    }

    // Combine along columns
    // Store the combined values in b2.
    for (i = 0; i < ncol; i++) {
        end_of_col = i + (nrow-1)*ncol;
        b2[i] = c1[i];
        combine_neighbors2(c1[i+ncol], b2+i);
        combine_neighbors2(c1[end_of_col], b2+i);
        for (j = i+ncol; j < end_of_col; j += ncol) {
            b2[j] = c1[j];
            combine_neighbors2(c1[j-ncol], b2+j);
            combine_neighbors2(c1[j+ncol], b2+j);
        }
        b2[end_of_col] = c1[end_of_col];
        combine_neighbors2(c1[i], b2 + end_of_col);
        combine_neighbors2(c1[end_of_col-ncol], b2 + end_of_col);
    }

    // Now loop over the board and advance it.
    for (i = 0; i < size; i++) {
        uint16_t num_alive = b2[i] & ALIVE_BITS;
        if (b1[i] & ALIVE) {
            // Note that if it's alive, it counts as its own neighbor.
            if (b1[i] & FROZEN || b2[i] & PRESERVING ||
                num_alive == 3 || num_alive == 4) {
                // copy the old cell
                b2[i] = b1[i];
            } else {
                // kill the cell
                b2[i] = 0;
            }
        } else {  // starts dead
            if (b1[i] & FROZEN || b2[i] & INHIBITING) {
                // copy the old cell
                b2[i] = b1[i];
            } else if (num_alive == 3) {
                // add a new live cell
                b2[i] = ALIVE |
                    ((b2[i] & (COLORS << 4)) >> 4) |
                    ((b2[i] & (DESTRUCTIBLE2 << 4)) >> 9);
            } else if (b2[i] & SPAWNING && random_float() < spawn_prob) {
                // add a spawned cell
                b2[i] = ALIVE | DESTRUCTIBLE |
                    ((b2[i] & (COLORS << 4)) >> 4);
            } else {
                // copy the old cell
                b2[i] = b1[i];
            }
        }
    }
}

