#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "constants.h"
#include "iset.h"
#include "gen_board.h"

#if 0
    #include <Python.h>
    #define PRINT(...) PySys_WriteStdout(__VA_ARGS__)
#else
    #define PRINT(...) ;
#endif

#define EMPTY_IDX  0
#define ALIVE_IDX  1
#define WALL_IDX  2
#define TREE_IDX  3
#define WEED_IDX  4
#define PREDATOR_IDX  5
#define ICECUBE_IDX  6
#define FOUNTAIN_IDX  7


static int16_t cell_type_array[8] = {
    0,
    ALIVE | DESTRUCTIBLE,
    FROZEN,
    FROZEN | ALIVE,
    FROZEN | ALIVE | PRESERVING | MOVABLE,
    FROZEN | ALIVE | INHIBITING | MOVABLE,
    FROZEN | PRESERVING | INHIBITING | MOVABLE,
    FROZEN | PRESERVING,
};


static int idx_for_cell_type(int16_t cell) {
    int idx = 0;
    if (cell & FROZEN) {
        if (cell & ALIVE) {
            if (cell & MOVABLE)
                idx = cell & PRESERVING ? WEED_IDX : PREDATOR_IDX;
            else
                idx = TREE_IDX;
        } else {  // frozen, not alive
            if (cell & PRESERVING)
                idx = cell & INHIBITING ? ICECUBE_IDX : FOUNTAIN_IDX;
            else
                idx = WALL_IDX;
        }
    } else {  // not frozen
        if (cell & ALIVE)
            idx = ALIVE_IDX;
        else
            idx = EMPTY_IDX;
    }
    return idx;
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
        temp[end_of_row] = array[end_of_row-1];
        temp[end_of_row] = array[start_of_row];
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


int gen_still_life(int16_t *board, int32_t *mask, int nrow, int ncol) {
    // The amount of indexing gymnastics used here is, admitedly, awful.
    // We could reduce it somewhat with more index division and modulos, but
    // that'll slow things down and won't make it that much more clear.
    int size = nrow*ncol;
    int n_alive[size];
    int n_preserved[size];
    int n_inhibited[size];
    int tmp_buffer[size];
    int total_area = 0;
    int totals[8];
    int num_iter;
    iset bad_idx = iset_alloc(size);

    // The following few should be passed in as parameters.
    int max_iter = size * 40;
    // max_iter = 1;
    double min_fill = 0.2;
    double temperature = 0.7;
    double beta = 1 / temperature;
    double cell_penalties[16] = {
        // intercept and slope of penalty
        0, 0,  // EMPTY (handled separately by min_fill)
        0, 0,  // ALIVE
        100, 40,  // WALL
        1, 30,  // TREE
        100, 100,  // WEED
        100, 100,  // PREDATOR
        100, 100,  // ICECUBE
        100, 0,  // FOUNTAIN
    };

    for (int k = 0; k < size; k++) {
        n_alive[k] = board[k] & ALIVE;
        n_preserved[k] = (board[k] & PRESERVING) >> PRESERVING_BIT;
        n_inhibited[k] = (board[k] & INHIBITING) >> INHIBITING_BIT;
    }
    wrapped_convolve(n_alive, tmp_buffer, nrow, ncol);
    wrapped_convolve(n_preserved, tmp_buffer, nrow, ncol);
    wrapped_convolve(n_inhibited, tmp_buffer, nrow, ncol);

    // Create a set of all cells that violate the still-life rules.
    memset(totals, 0, sizeof(totals));
    for (int k = 0; k < size; k++) {
        int16_t val = board[k];
        // Assume that there are no violations that are masked out,
        // although we'll make sure not to create new ones in the masked area.
        if (!mask[k]) continue;
        total_area++;
        totals[idx_for_cell_type(val)]++;
        if (val & FROZEN) continue;
        else if (val & ALIVE) {
            if (!n_preserved[k] && (n_alive[k] < 3 || n_alive[k] > 4))
                iset_add(&bad_idx, k);
        } else {
            if (!n_inhibited[k] && n_alive[k] == 3)
                iset_add(&bad_idx, k);
        }
    }

    for (num_iter = 0; num_iter < max_iter; num_iter++) {
        int not_empty = total_area - totals[0];
        if (bad_idx.size == 0 && not_empty >= total_area * min_fill) {
            break;  // Success!
        }

        // Sample a point
        int k0, r0, c0;
        k0 = iset_sample(&bad_idx);
        r0 = k0 / ncol;
        c0 = k0 % ncol;
        // build the 5x5 neighborhood surrounding the point.
        int dxn2 = (c0 >= 2     ? -2 : -2 + ncol);
        int dxn1 = (c0 >= 1     ? -1 : -1 + ncol);
        int dxp1 = (c0 < ncol-1 ? +1 : +1 - ncol);
        int dxp2 = (c0 < ncol-2 ? +2 : +2 - ncol);
        int dyn2 = (r0 >= 2     ? -2 : -2 + nrow) * ncol;
        int dyn1 = (r0 >= 1     ? -1 : -1 + nrow) * ncol;
        int dyp1 = (r0 < nrow-1 ? +1 : +1 - nrow) * ncol;
        int dyp2 = (r0 < nrow-2 ? +2 : +2 - nrow) * ncol;
        int i1[25] = {
            k0 + dxn2 + dyn2,
            k0 + dxn1 + dyn2,
            k0        + dyn2,
            k0 + dxp1 + dyn2,
            k0 + dxp2 + dyn2,
            k0 + dxn2 + dyn1,
            k0 + dxn1 + dyn1,
            k0        + dyn1,
            k0 + dxp1 + dyn1,
            k0 + dxp2 + dyn1,
            k0 + dxn2,
            k0 + dxn1,
            k0       ,
            k0 + dxp1,
            k0 + dxp2,
            k0 + dxn2 + dyp1,
            k0 + dxn1 + dyp1,
            k0        + dyp1,
            k0 + dxp1 + dyp1,
            k0 + dxp2 + dyp1,
            k0 + dxn2 + dyp2,
            k0 + dxn1 + dyp2,
            k0        + dyp2,
            k0 + dxp1 + dyp2,
            k0 + dxp2 + dyp2,
        };
        int i2[9] = {6, 7, 8, 11, 12, 13, 16, 17, 18};
        int i3[8] = {-6, -5, -4, -1, +1, +4, +5, +6};
        int i4[9] = {-6, -5, -4, -1, 0, +1, +4, +5, +6};

        // Now for each cell in the Moore neighborhood, count up how many
        // violations there'd be if that cell were each of the different cell
        // types: empty, alive, crate, tree, weed, predator, fountain, ice cube
        int violations[72];
        memset(violations, 0, sizeof(violations));
        for (int j2 = 0; j2 < 9; j2++) {
            int k2 = i1[i2[j2]];
            if (!mask[k2]) continue;  // don't need to calculate violations
            int16_t current_cell = board[k2];
            int is_alive = current_cell & ALIVE;
            int is_preserving = (current_cell & PRESERVING) >> PRESERVING_BIT;
            int is_inhibiting = (current_cell & INHIBITING) >> INHIBITING_BIT;
            int num_alive, num_preserving, num_inhibiting;

            // Add violations due to this cell's effect on neighbors
            for (int j3 = 0; j3 < 8; j3++) {
                int k3 = i1[i2[j2] + i3[j3]];
                int16_t neighbor_cell = board[k3];
                num_alive = n_alive[k3] - is_alive;
                num_preserving = n_preserved[k3] - is_preserving;
                num_inhibiting = n_inhibited[k3] - is_inhibiting;
                if (neighbor_cell & FROZEN) continue;
                else if (neighbor_cell & ALIVE && num_preserving == 0) {
                    if (num_alive >= 4) {
                        // note that the preserving cells
                        // (ice cube, weed, fountain) don't have violations
                        // for living neighbors
                        violations[j2*8 + EMPTY_IDX] += num_alive - 4;
                        violations[j2*8 + WALL_IDX] += num_alive - 4;
                        violations[j2*8 + PREDATOR_IDX] += num_alive - 4;
                        violations[j2*8 + ALIVE_IDX] += num_alive - 3;
                        violations[j2*8 + TREE_IDX] += num_alive - 3;
                    } else if (num_alive <= 2){
                        violations[j2*8 + EMPTY_IDX] += 3 - num_alive;
                        violations[j2*8 + WALL_IDX] += 3 - num_alive;
                        violations[j2*8 + PREDATOR_IDX] += 3 - num_alive;
                        violations[j2*8 + ALIVE_IDX] += 2 - num_alive;
                        violations[j2*8 + TREE_IDX] += 2 - num_alive;
                    }
                } else if (!(neighbor_cell & ALIVE) && num_inhibiting == 0) {
                    // Likewise, inhibiting cells (predator and ice cube)
                    // don't have violations for dead neighbors
                    violations[j2*8 + EMPTY_IDX] += num_alive != 4;
                    violations[j2*8 + WALL_IDX] += num_alive != 4;
                    violations[j2*8 + FOUNTAIN_IDX] += num_alive != 3;
                    violations[j2*8 + ALIVE_IDX] += num_alive != 3;
                    violations[j2*8 + WEED_IDX] += num_alive != 3;
                    violations[j2*8 + TREE_IDX] += num_alive != 3;
                }
            }

            // Add violations for this cell by itself
            // Only alive and empty cells are non-frozen,
            // so only those receive violations.
            num_alive = n_alive[k2] - is_alive;
            num_preserving = n_preserved[k2] - is_preserving;
            num_inhibiting = n_inhibited[k2] - is_inhibiting;
            if (num_alive < 3) {
                violations[j2*8 + ALIVE_IDX] += 2 - num_alive;
            } else if (num_alive > 3) {
                violations[j2*8 + ALIVE_IDX] += num_alive - 3;
            } else {
                violations[j2*8 + EMPTY_IDX] += 1;
            }
        }

        {  // block cell picking
            // Convert violations to probabilities, and use those to switch
            // one of the cells.
            int j2, j3, j4, k2;
            int16_t old_cell, new_cell;
            double probs[72];
            double *penalties = probs;  // alias; we only need one at a time
            double cum_prob = 0.0;
            double min_penalty = 1000;
            double cell_penalties2[8];

            {
                // Special penalty for empty cell
                // Starts at 2 (i.e., empty cell has same total penalty
                // as a live cell with no neighbors), but quickly drops
                // to zero once we approach min_fill.
                double t = not_empty / (total_area * min_fill);
                cell_penalties2[0] = t < 0.9 ? 2.0 : t < 1 ? 20 * (1 - t) : 0;
            }
            for (j3 = 1; j3 < 8; j3++) {
                double t = totals[j3] / (not_empty + 1.0);
                cell_penalties2[j3] = cell_penalties[j3*2] + t * cell_penalties[j3*2+1];
            }

            for (j2 = 0, j4 = 0; j2 < 9; j2++) {
                k2 = i1[i2[j2]];
                old_cell = board[k2];
                for (j3 = 0; j3 < 8; j3++, j4++) {
                    // Add penalties to the different cell types
                    penalties[j4] = violations[j4] + cell_penalties2[j3];
                    new_cell = cell_type_array[j3];
                    if (min_penalty > penalties[j4] && mask[k2] && new_cell != old_cell) {
                        min_penalty = penalties[j4];
                    }
                }
            }
            for (j2 = 0, j4 = 0; j2 < 9; j2++) {
                k2 = i1[i2[j2]];
                old_cell = board[k2];
                for (j3 = 0; j3 < 8; j3++, j4++) {
                    new_cell = cell_type_array[j3];
                    double x = (min_penalty - penalties[j4]) * beta;
                    probs[j4] = exp(x);
                    probs[j4] *= mask[k2] * (new_cell != old_cell);
                    cum_prob += probs[j4];
                }
            }
            if (cum_prob == 0.0) {
                // should never get here...
                // Just drop this as a bad index and keep on our merry way.
                iset_discard(&bad_idx, k0);
                continue;
            }

            // Pick a cell!
            double target_prob = rand() * cum_prob / RAND_MAX;
            cum_prob = 0.0;
            for (j2 = 0, j4 = 0; j2 < 9; j2++) {
                for (j3 = 0; j3 < 8; j3++, j4++) {
                    cum_prob += probs[j4];
                    if (target_prob <= cum_prob) {
                        goto swap_cell; // Pick this cell!
                    }
                }
            }
            { // no cell selected (didn't hit the goto)
                PRINT(
                   "ERROR! target_prob > cum_prob: %5g; %5g\n",
                   target_prob, cum_prob);
                iset_free(&bad_idx);
                return -1;
            }

            // Swap in the new cell
            swap_cell:
            k2 = i1[i2[j2]];
            old_cell = board[k2];
            new_cell = cell_type_array[j3];
            totals[idx_for_cell_type(new_cell)]++;
            totals[idx_for_cell_type(old_cell)]--;
            board[k2] = new_cell;
            PRINT("swap (%i %i): %i -> %i\n",
                k2 / ncol, k2 % ncol,
                idx_for_cell_type(old_cell), idx_for_cell_type(new_cell));

            // Adjust the neighbors and bad_idx
            int delta_alive = (new_cell & ALIVE) - (old_cell & ALIVE);
            int delta_preserve = ((new_cell & PRESERVING) >> PRESERVING_BIT) -
                ((old_cell & PRESERVING) >> PRESERVING_BIT);
            int delta_inhibit = ((new_cell & INHIBITING) >> INHIBITING_BIT) -
                ((old_cell & INHIBITING) >> INHIBITING_BIT);
            for (j4 = 0; j4 < 9; j4++) {
                int k3 = i1[i2[j2] + i4[j4]];
                int16_t neighbor = board[k3];
                n_alive[k3] += delta_alive;
                n_preserved[k3] += delta_preserve;
                n_inhibited[k3] += delta_inhibit;
                if (neighbor & FROZEN) {
                    iset_discard(&bad_idx, k3);
                } else if (neighbor & ALIVE) {
                    if (n_preserved[k3]) {
                        iset_discard(&bad_idx, k3);
                    } else if (n_alive[k3] == 3 || n_alive[k3] == 4) {
                        iset_discard(&bad_idx, k3);
                    } else {
                        iset_add(&bad_idx, k3);
                    }
                } else {  // not alive, not frozen
                    if (n_inhibited[k3]) {
                        iset_discard(&bad_idx, k3);
                    } else if (n_alive[k3] != 3) {
                        iset_discard(&bad_idx, k3);
                    } else {
                        iset_add(&bad_idx, k3);
                    }
                }
           }  // end loop over neighbors
        }  // end block cell picking
    }  // end outer loop

    iset_free(&bad_idx);
    PRINT("Iterations: %i/%i\n", num_iter, max_iter);
    return num_iter == max_iter ? -1 : 0;
}
