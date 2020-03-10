#include <stdint.h>
#include <string.h>
#include <math.h>
#include "constants.h"
#include "iset.h"
#include "gen_board.h"
#include "random.h"

#if 0
    #include <Python.h>
    #define PRINT(...) PySys_WriteStdout(__VA_ARGS__)
    //#define PRINT(...) printf(__VA_ARGS__)
#else
    #define PRINT(...) do {} while (0);
#endif

#define EMPTY_IDX  0
#define WALL_IDX  1
#define ALIVE_IDX  2
#define TREE_IDX  3


static uint16_t cell_type_array[4] = {
    0,
    FROZEN,
    ALIVE | DESTRUCTIBLE,
    FROZEN | ALIVE,
};


typedef struct {
    int oscillations;
    int violations;
} swap_cells_t;


typedef struct {
    int x1;
    int y1;
    int x2;
    int y2;
} bounds_t;


static int idx_for_cell_type(uint16_t cell) {
    // only consider empty, alive, wall and tree
    // alive gets stored in second bit, frozen in first.
    return
        (((cell & ALIVE) >> ALIVE_BIT) << 1) |
        ((cell & FROZEN) >> FROZEN_BIT);
}


static void wrapped_convolve(int *array, int *temp, int nrow, int ncol) {
    // Do a 3x3 convolution on the array in place.
    int i, j, start_of_row, end_of_row, end_of_col;

    // Combine along rows
    for (i = 0; i < nrow; i++) {
        // Wrap at start of row
        start_of_row = i * ncol;
        end_of_row = start_of_row + ncol - 1;
        temp[start_of_row] = array[start_of_row];
        temp[start_of_row] += array[start_of_row+1];
        temp[start_of_row] += array[end_of_row];
        // Loop over middle
        for (j = start_of_row + 1; j < end_of_row; j++) {
            temp[j] = array[j];
            temp[j] += array[j+1];
            temp[j] += array[j-1];
        }
        // Wrap at end of row
        temp[end_of_row] = array[end_of_row];
        temp[end_of_row] += array[end_of_row-1];
        temp[end_of_row] += array[start_of_row];
    }

    // Combine along columns
    for (i = 0; i < ncol; i++) {
        end_of_col = i + (nrow-1)*ncol;
        array[i] = temp[i];
        array[i] += temp[i+ncol];
        array[i] += temp[end_of_col];
        for (j = i+ncol; j < end_of_col; j += ncol) {
            array[j] = temp[j];
            array[j] += temp[j-ncol];
            array[j] += temp[j+ncol];
        }
        array[end_of_col] = temp[end_of_col];
        array[end_of_col] += temp[i];
        array[end_of_col] += temp[end_of_col-ncol];
    }
}


/*static int calc_interior_area(uint32_t *mask, int nrow, int ncol) {
    // Quick loop to calculate the number of unmasked cells that do not
    // border masked cells.
    int total = 0;
    for (int i = 0, k = 0; i < nrow; i++) {
        int dy1 = (i > 0 ? -1 : nrow - 1) * ncol;
        int dy2 = (i < nrow - 1 ? +1 : 1 - nrow) * ncol;
        for (int j = 0; j < ncol; j++, k++) {
            int dx1 = (j > 0 ? -1 : ncol - 1);
            int dx2 = (j < ncol - 1 ? +1 : 1 - ncol);
            if ((1 & mask[k+dy1]) && (1 & mask[k+dy2]) &&
                (1 & mask[k+dx1]) && (1 & mask[k+dx2]) && (1 & mask[k])) {
                total++;
            }
        }
    }
    return total;
}*/


static int _idx(int i, int j, int k, board_shape_t s) {
    // The bounds checking here will generally be pretty redundant, but it
    // makes the code much easier to read. Can try to optimize in a later step.
    while (j < 0) j += s.rows;
    while (k < 0) k += s.cols;
    while (j >= s.rows) j -= s.rows;
    while (k >= s.cols) k -= s.cols;
    return k + (j + i * s.rows) * s.cols;
}


static int swap_single_cell(
        uint16_t *board, int *neighbors, board_shape_t board_shape,
        int layer, int row, int col, uint16_t new_cell) {
    // Swap out a single cell type and update the neighboring cells.
    // Returns:
    //     0 if new cell is the same as old cell
    //     1 if only FROZEN bit switched
    //     2 if ALIVE bit switched
    int i0 = _idx(layer, row, col, board_shape);
    uint16_t old_cell = board[i0];
    if (old_cell == new_cell) {
        return 0;
    }

    board[i0] = new_cell;
    int delta_alive = (new_cell & ALIVE) - (old_cell & ALIVE);

    if (delta_alive) {
        for (int r = row-1; r <= row+1; r++) {
            for (int c = col-1; c <= col+1; c++) {
                neighbors[_idx(layer, r, c, board_shape)] += delta_alive;
            }
        }
        return 2;
    } else {
        return 1;
    }
}


static int check_for_violation(uint16_t src, uint16_t dst, int neighbors) {
    int rval;
    if (src & FROZEN) {
        rval = src != dst;
    } else if (src & ALIVE) {
        rval = (neighbors == 3 || neighbors == 4) ^ ((dst & ALIVE) != 0);
    } else {
        rval = (neighbors == 3) ^ ((dst & ALIVE) != 0);
    }
    // PRINT("violations(%i, %i, %i) -> %i\n", src, dst, neighbors, rval);
    return rval;
}

/*static char print_cell(int16_t cell) {
    // debugging only
    switch (cell) {
        case 0:
            return '.';
        case ALIVE:
            return 'Z';
        case ALIVE | DESTRUCTIBLE:
            return 'z';
        case FROZEN:
            return '#';
        case ALIVE | FROZEN:
            return 'T';
        default:
            return '?';
    }
}*/


static swap_cells_t swap_cells(
        uint16_t *board, int *neighbors, int *violations, int *oscillations, int *mask,
        board_shape_t board_shape, int row, int col, uint16_t new_cell,
        iset *bad_idx) {

    swap_cells_t delta_swap = {0, 0};
    bounds_t area_of_effect = {col, row, col, row};
    int layer_size = board_shape.rows * board_shape.cols;
    // int start_cell = board[_idx(0, row, col, board_shape)]; // debug only!

    int did_swap = swap_single_cell(
        board, neighbors, board_shape, 0, row, col, new_cell);
    switch (did_swap) {
        case 0: return delta_swap;
        case 1: break;  // swap only changed frozen status
        case 2:
            area_of_effect.x1--;
            area_of_effect.y1--;
            area_of_effect.x2++;
            area_of_effect.y2++;
    }

    for (int layer=1; layer < board_shape.depth; layer++) {
        // Evolve the next layer outwards
        int swap_types = 0;
        for (int r = area_of_effect.y1; r <= area_of_effect.y2; r++) {
            for (int c = area_of_effect.x1; c <= area_of_effect.x2; c++) {
                int i1 = _idx(layer-1, r, c, board_shape);
                uint16_t b1 = board[i1], b2;
                int n1 = neighbors[i1];
                if (b1 & FROZEN) {
                    b2 = b1;
                } else if (b1 & ALIVE) {
                    b2 = (n1 == 3 || n1 == 4) ? b1 : 0;
                } else {
                    b2 = (n1 == 3) ? ALIVE : b1;
                }
                did_swap = swap_single_cell(
                        board, neighbors, board_shape, layer, r, c, b2);
                swap_types |= did_swap;
                if (did_swap) {
                    if (c == area_of_effect.x1) area_of_effect.x1 -= 1;
                    if (c == area_of_effect.x2) area_of_effect.x2 += 1;
                    if (r == area_of_effect.y1) area_of_effect.y1 -= 1;
                    if (r == area_of_effect.y2) area_of_effect.y2 += 1;
                }
            }
        }
        if (!swap_types) break;  // end early; nothing to do.
    }

    // Loop through the updated area and update the violations and oscillations
    int total_violations = 0; // Debug only!
    for (int r = area_of_effect.y1; r <= area_of_effect.y2; r++) {
        for (int c = area_of_effect.x1; c <= area_of_effect.x2; c++) {
            int i1 = _idx(0, r, c, board_shape);
            int i2 = i1;
            int _oscillations, _violations;
            // oscillations are given by storing dead bits in the usual ALIVE
            // spot, and live bits in the next bit over. If both are present
            // the total should be 3 * ALIVE.
            const int is_osc = 3 * ALIVE;
            uint16_t b1 = board[i1], b2;
            if (b1 & FROZEN) {
                _oscillations = 0;
                _violations = 0;
            } else {
                // Got to loop through layers to check for oscillations
                b2 = b1;
                _oscillations = (b1 & ALIVE) + ALIVE;
                for (int layer=1; layer < board_shape.depth; layer++) {
                    i2 += layer_size;
                    b2 = board[i2];
                    _oscillations |= (b2 & ALIVE) + ALIVE;
                }
                _violations = check_for_violation(b2, b1, neighbors[i2]);
            }
            if (_oscillations == is_osc && !(mask[i1] & CAN_OSCILLATE_MASK)) {
                _violations += 1;
            }
            delta_swap.violations += _violations - violations[i1];
            delta_swap.oscillations += (_oscillations == is_osc);
            delta_swap.oscillations -= (oscillations[i1] == is_osc);
            violations[i1] = _violations;
            oscillations[i1] = _oscillations;
            total_violations += _violations;
            if (bad_idx) {
                if (_violations && (mask[i1] & INCLUDE_VIOLATIONS_MASK)) {
                    iset_add(bad_idx, i1);
                } else {
                    iset_discard(bad_idx, i1);
                }
            }
        }
    }

    // Print out the area of effect (for debugging only)
    /*PRINT("\nswap cells: %c -> %c\n", print_cell(start_cell), print_cell(new_cell));
    PRINT("are of effect: %i, %i, %i, %i\n", area_of_effect.x1, area_of_effect.y1, area_of_effect.x2, area_of_effect.y2);
    PRINT("violations=%i\n", total_violations);
    for (int layer=0; layer < board_shape.depth; layer++) {
        PRINT("t = %i\n", layer);
        for (int r=row-2; r <= row+2; r++) {
            PRINT("   ");
            for (int c=col-2; c <= col+2; c++) {
                PRINT(" %c", print_cell(board[_idx(layer, r, c, board_shape)]));
            }
            PRINT("  | ");
            for (int c=col-2; c <= col+2; c++) {
                PRINT(" %i", neighbors[_idx(layer, r, c, board_shape)]);
            }
            PRINT("  | ");
            for (int c=col-2; c <= col+2; c++) {
                PRINT(" %i", violations[_idx(0, r, c, board_shape)]);
            }
            PRINT("  | ");
            for (int c=col-2; c <= col+2; c++) {
                PRINT(" %i", oscillations[_idx(0, r, c, board_shape)]);
            }
            PRINT("\n");
        }
    }*/

    return delta_swap;
}


int gen_pattern(
        uint16_t *board, int32_t *mask, int32_t *seeds, board_shape_t shape,
        double rel_max_iter, double rel_min_fill, double temperature, double osc_bonus,
        double *cell_penalties) {
    PRINT("\nStarting the oscillator! (%i %i %i)\n", shape.depth, shape.rows, shape.cols);

    // Assume that the board is already filled out in multiple layers.
    // Still need to build the violations, oscillations, and neighbors though.
    int layer_size = shape.rows * shape.cols;
    int board_size = shape.depth * layer_size;
    int last_layer_idx = board_size - layer_size;
    int total_area = 0;

    int neighbors[board_size];
    int oscillations[layer_size];
    int violations[layer_size];
    int totals[4] = {0, 0, 0, 0};
    iset bad_idx = iset_alloc(layer_size);
    iset unmasked_idx = iset_alloc(layer_size);
    iset seeds_idx = iset_alloc(layer_size);

    for (int i=0; i < board_size; i++) {
        neighbors[i] = board[i] & ALIVE;
    }
    memset(oscillations, 0, sizeof(oscillations));
    for (int i=0; i < shape.depth; i++) {
        for (int k=0; k < layer_size; k++) {
            oscillations[k] |= (board[k + i*layer_size] & ALIVE) + ALIVE;
        }
    }

    for (int i=0; i < shape.depth; i++) {
        int temp[layer_size];
        wrapped_convolve(neighbors + i*layer_size, temp, shape.rows, shape.cols);
    }

    for (int i=0; i < layer_size; i++) {
        violations[i] = check_for_violation(
            board[i + last_layer_idx], board[i], neighbors[i + last_layer_idx]);
        if (seeds[i]) {
            iset_add(&seeds_idx, i);
        }
        if (violations[i] && (mask[i] & INCLUDE_VIOLATIONS_MASK)) {
            iset_add(&bad_idx, i);
        }
        if (mask[i] & NEW_CELL_MASK) {
            iset_add(&unmasked_idx, i);
            total_area++;
            int cell_type_idx = idx_for_cell_type(board[i]);
            totals[cell_type_idx]++;
        }
    }

    // Calculate some constants for the loop
    int max_iter = rel_max_iter * total_area * shape.depth;
    //int interior_area = calc_interior_area(mask, shape.rows, shape.cols);
    //double effective_area = 0.75 * interior_area + 0.25 * total_area;
    double min_fill = rel_min_fill * total_area;
    PRINT("Total area: %i; ", total_area);
    // PRINT("Interior area: %i\n", interior_area);
    // if (interior_area < 2) {
    //     PRINT("AREA TOO SMALL!\n");
    //     return AREA_TOO_SMALL_ERROR;
    // }

    PRINT("\nSTARTING THE LOOP\n\n");

    // And start the loop!
    int num_iter;
    int not_empty = 0;
    for (num_iter=0; num_iter < max_iter; num_iter++) {
        not_empty = total_area - totals[EMPTY_IDX];
        if (bad_idx.size == 0 && not_empty >= min_fill) {
            break;  // Success!
        }

        // Sample a point
        int k0, r0, c0;
        k0 = iset_sample(
            bad_idx.size > 0 ? &bad_idx :
            seeds_idx.size > 0 ? &seeds_idx :
            &unmasked_idx);
        // Sample from each seed no more than once.
        iset_discard(&seeds_idx, k0);
        r0 = k0 / shape.cols;
        c0 = k0 % shape.cols;

        // Figure out what the current penalties are for different cell types.
        double cell_penalties2[4];
        {
            // Special penalty for empty cell
            // Starts at 2 (i.e., empty cell has same total penalty
            // as a live cell with no neighbors), but quickly drops
            // to zero once we approach min_fill.
            double t = not_empty / min_fill;
            cell_penalties2[0] = t < 0.9 ? 2.0 : t < 1 ? 20 * (1 - t) : 0;
        }
        for (int j3 = 1; j3 < 4; j3++) {
            double t = totals[j3] / (not_empty + 1.0);
            double base = cell_penalties[j3*2];
            double slope = cell_penalties[j3*2+1];
            cell_penalties2[j3] = base + t*slope;
        }

        double beta = 1.0 / temperature;
        int neighborhood_size = (2*shape.depth+1) * (2*shape.depth+1);
        double log_probs[4 * neighborhood_size];
        uint16_t cell_types[4 * neighborhood_size];
        int switched_idx[4 * neighborhood_size];
        double max_log_prob = -1e100;

        // Try switching each cell in the target's extended neighborhood.
        int num_switched = 0;
        for (int r = r0 - shape.depth; r <= r0 + shape.depth; r++) {
            for (int c = c0 - shape.depth; c <= c0 + shape.depth; c++) {
                //PRINT("\nr,c = %i,%i\n", r, c);
                int i1 = _idx(0, r, c, shape);
                if (!(mask[i1] & NEW_CELL_MASK)) continue;
                uint16_t current_cell = board[i1];
                int start_idx = idx_for_cell_type(current_cell) + 1;
                int delta_violations = 0;
                int delta_oscillations = 0;
                for (int j = start_idx; j < start_idx+3; j++) {
                    uint16_t target_type = cell_type_array[j & 3];
                    swap_cells_t delta = swap_cells(
                        board, neighbors, violations, oscillations, mask,
                        shape, r, c, target_type, NULL);
                    delta_violations += delta.violations;
                    delta_oscillations += delta.oscillations;
                    log_probs[num_switched] = delta_violations;
                    log_probs[num_switched] -= osc_bonus * delta_oscillations;
                    log_probs[num_switched] += cell_penalties2[j & 3];
                    log_probs[num_switched] *= -beta;
                    if (max_log_prob < log_probs[num_switched]) {
                        max_log_prob = log_probs[num_switched];
                    }
                    cell_types[num_switched] = target_type;
                    switched_idx[num_switched] = i1;
                    num_switched++;
                }
                // Then switch back to the old cell type.
                //PRINT("swap back!\n");
                swap_cells(
                    board, neighbors, violations, oscillations, mask,
                    shape, r, c, current_cell, NULL);
            }
        }

        // Go through all of the switched cells and pick one.
        double total_prob = 0.0;
        double *cum_probs = log_probs;  // change from log prob to cumulative prob
        for (int k=0; k < num_switched; k++) {
            // PRINT("logprob: %c -> %f\n", print_cell(cell_types[k]), log_probs[k]);
            total_prob += exp(log_probs[k] - max_log_prob);
            cum_probs[k] = total_prob;
        }
        double target_prob = random_float() * total_prob;
        int k;
        for (k=0; k < num_switched; k++) {
            if (cum_probs[k] > target_prob) {
                // Pick this cell!
                int cell_idx = switched_idx[k];
                uint16_t old_cell = board[cell_idx];
                uint16_t new_cell = cell_types[k];
                // PRINT("PICK: %c -> %c; i=(%i,%i); m=%i\n",
                //     print_cell(old_cell), print_cell(new_cell),
                //     cell_idx / shape.cols, cell_idx % shape.cols, mask[cell_idx]);
                swap_cells(
                    board, neighbors, violations, oscillations, mask,
                    shape, cell_idx / shape.cols, cell_idx % shape.cols,
                    new_cell, &bad_idx);
                totals[idx_for_cell_type(old_cell)]--;
                totals[idx_for_cell_type(new_cell)]++;
                break;
            }
        }
        if (k >= num_switched) {
            PRINT("no cell picked... %g %g %g %i\n",
                target_prob, total_prob, max_log_prob, num_switched);
        }
    }

    iset_free(&bad_idx);
    iset_free(&unmasked_idx);
    iset_free(&seeds_idx);
    PRINT("Iterations: %i/%i\n", num_iter, max_iter);
    PRINT("Num alive: %i/%i\n", totals[ALIVE_IDX], total_area);
    return num_iter == max_iter ? MAX_ITER_ERROR : 0;
}

