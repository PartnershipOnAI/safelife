#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "constants.h"
#include "iset.h"
#include "gen_board.h"

#if 1
    #include <Python.h>
    //#define PRINT(...) PySys_WriteStdout(__VA_ARGS__)
    #define PRINT(...) printf(__VA_ARGS__)
#else
    #define PRINT(...) do {} while (0);
#endif

#define EMPTY_IDX  0
#define ALIVE_IDX  2
#define WALL_IDX  1
#define TREE_IDX  3
#define WEED_IDX  4
#define PREDATOR_IDX  5
#define ICECUBE_IDX  6
#define FOUNTAIN_IDX  7


static int16_t cell_type_array[8] = {
    0,
    FROZEN,
    ALIVE | DESTRUCTIBLE,
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


static int idx_for_cell_type2(int16_t cell) {
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


static int calc_interior_area(int32_t *mask, int nrow, int ncol) {
    // Quick loop to calculate the number of unmasked cells that do not
    // border masked cells.
    int total = 0;
    for (int i = 0, k = 0; i < nrow; i++) {
        int dy1 = (i > 0 ? -1 : nrow - 1) * ncol;
        int dy2 = (i < nrow - 1 ? +1 : 1 - nrow) * ncol;
        for (int j = 0; j < ncol; j++, k++) {
            int dx1 = (j > 0 ? -1 : ncol - 1);
            int dx2 = (j < ncol - 1 ? +1 : 1 - ncol);
            if (mask[k+dy1] && mask[k+dy2] &&
                mask[k+dx1] && mask[k+dx2] && mask[k]) {
                total++;
            }
        }
    }
    return total;
}


int gen_still_life(
        int16_t *board, int32_t *mask, int32_t *seeds, int nrow, int ncol,
        double rel_max_iter, double rel_min_fill, double temperature,
        double *cell_penalties)
{
    // The amount of indexing gymnastics used here is, admitedly, awful.
    // We could reduce it somewhat with more index division and modulos, but
    // that'll slow things down and won't actually make it that much more clear.
    int size = nrow*ncol;
    int n_alive[size];
    int n_preserved[size];
    int n_inhibited[size];
    int tmp_buffer[size];
    int total_area = 0;
    int totals[8];
    int num_iter;
    iset bad_idx = iset_alloc(size);
    iset unmasked_idx = iset_alloc(size);
    iset seeds_idx = iset_alloc(size);
    double beta = 1 / (temperature > 0.01 ? temperature : 0.01);

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
        if (seeds[k]) iset_add(&seeds_idx, k);
        if (!mask[k]) continue;
        total_area++;
        totals[idx_for_cell_type(val)]++;
        iset_add(&unmasked_idx, k);
        if (val & FROZEN) continue;
        else if (val & ALIVE) {
            if (!n_preserved[k] && (n_alive[k] < 3 || n_alive[k] > 4))
                iset_add(&bad_idx, k);
        } else {
            if (!n_inhibited[k] && n_alive[k] == 3)
                iset_add(&bad_idx, k);
        }
    }

    int max_iter = rel_max_iter * total_area;
    int interior_area = calc_interior_area(mask, nrow, ncol);
    double effective_area = 0.75 * interior_area + 0.25 * total_area;
    double min_fill = rel_min_fill * effective_area;
    PRINT("Total area: %i; ", total_area);
    PRINT("Interior area: %i\n", interior_area);
    if (interior_area < 2) {
        return AREA_TOO_SMALL_ERROR;
    }

    for (num_iter = 0; num_iter < max_iter; num_iter++) {
        int not_empty = total_area - totals[0];
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
        static int i2[9] = {6, 7, 8, 11, 12, 13, 16, 17, 18};
        static int i3[8] = {-6, -5, -4, -1, +1, +4, +5, +6};
        static int i4[9] = {-6, -5, -4, -1, 0, +1, +4, +5, +6};

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
                    int not2or3 = (num_alive < 2) || (num_alive > 3);
                    int not3or4 = (num_alive < 3) || (num_alive > 4);
                    violations[j2*8 + EMPTY_IDX] += not3or4;
                    violations[j2*8 + WALL_IDX] += not3or4;
                    violations[j2*8 + PREDATOR_IDX] += not2or3;
                    violations[j2*8 + ALIVE_IDX] += not2or3;
                    violations[j2*8 + TREE_IDX] += not2or3;
                    /*if (num_alive >= 4) {
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
                    }*/
                } else if (!(neighbor_cell & ALIVE) && num_inhibiting == 0) {
                    // Likewise, inhibiting cells (predator and ice cube)
                    // don't have violations for dead neighbors
                    int not3 = num_alive != 3;
                    int not4 = num_alive != 4;
                    violations[j2*8 + EMPTY_IDX] += not4;
                    violations[j2*8 + WALL_IDX] += not4;
                    violations[j2*8 + FOUNTAIN_IDX] += not3;
                    violations[j2*8 + ALIVE_IDX] += not3;
                    violations[j2*8 + WEED_IDX] += not3;
                    violations[j2*8 + TREE_IDX] += not3;
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
            double cprob1 = 0.0, cprob2 = 0.0;
            double min_penalty = 1000;
            double cell_penalties2[8];
            int num_masked = 0;

            {
                // Special penalty for empty cell
                // Starts at 2 (i.e., empty cell has same total penalty
                // as a live cell with no neighbors), but quickly drops
                // to zero once we approach min_fill.
                double t = not_empty / min_fill;
                cell_penalties2[0] = t < 0.9 ? 2.0 : t < 1 ? 20 * (1 - t) : 0;
            }
            for (j3 = 1; j3 < 8; j3++) {
                double t = totals[j3] / (not_empty + 1.0);
                double base = cell_penalties[j3*2];
                double slope = cell_penalties[j3*2+1];
                cell_penalties2[j3] = base + t*slope;
            }

            for (j2 = 0, j4 = 0; j2 < 9; j2++) {
                k2 = i1[i2[j2]];
                old_cell = board[k2];
                if (!mask[k2]) {
                    num_masked++;
                    j4 += 8;
                    continue;
                }
                for (j3 = 0; j3 < 8; j3++, j4++) {
                    // Add penalties to the different cell types
                    penalties[j4] = violations[j4] + cell_penalties2[j3];
                    new_cell = cell_type_array[j3];
                    if (min_penalty > penalties[j4] && new_cell != old_cell) {
                        min_penalty = penalties[j4];
                    }
                }
            }
            if (num_masked == 9) {
                // Everything is masked out. Just go to the next point.
                iset_discard(&bad_idx, k0);
                continue;
            }
            for (j2 = 0, j4 = 0; j2 < 9; j2++) {
                k2 = i1[i2[j2]];
                old_cell = board[k2];
                for (j3 = 0; j3 < 8; j3++, j4++) {
                    new_cell = cell_type_array[j3];
                    probs[j4] = mask[k2] && new_cell != old_cell ?
                        exp((min_penalty - penalties[j4]) * beta) : 0.0;
                    cprob1 += probs[j4];
                }
            }
            if (cprob1 == 0.0) {
                // should never get here...
                // Just drop this as a bad index and keep on our merry way.
                iset_discard(&bad_idx, k0);
                continue;
            }

            // Pick a cell!
            double target_prob = rand() * cprob1 / RAND_MAX;
            for (j2 = 0, j4 = 0; j2 < 9; j2++) {
                for (j3 = 0; j3 < 8; j3++, j4++) {
                    cprob2 += probs[j4];
                    if (target_prob <= cprob2) {
                        goto swap_cell; // Pick this cell!
                    }
                }
            }
            { // no cell selected (didn't hit the goto)
                PRINT(
                   "ERROR! target_prob > cum_prob: %0.5g; %0.5g %0.5g\n",
                   target_prob, cprob1, cprob2);
                PRINT("k0 = %i, %i\n \n", k0/ncol, k0%ncol);
                iset_free(&bad_idx);
                iset_free(&unmasked_idx);
                iset_free(&seeds_idx);
                return PROBABILITY_ERROR;
            }

        swap_cell: // Swap in the new cell
            k2 = i1[i2[j2]];
            old_cell = board[k2];
            new_cell = cell_type_array[j3];
            totals[idx_for_cell_type(new_cell)]++;
            totals[idx_for_cell_type(old_cell)]--;
            board[k2] = new_cell;

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
    iset_free(&unmasked_idx);
    iset_free(&seeds_idx);
    PRINT("Iterations: %i/%i\n", num_iter, max_iter);
    PRINT("Num alive: %i/%i\n", totals[ALIVE_IDX], total_area);
    return num_iter == max_iter ? MAX_ITER_ERROR : 0;
}


// -----------------------------------------


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
        int16_t *board, int *neighbors, board_shape_t board_shape,
        int layer, int row, int col, int16_t new_cell) {
    // Swap out a single cell type and update the neighboring cells.
    // Returns:
    //     0 if new cell is the same as old cell
    //     1 if only FROZEN bit switched
    //     2 if ALIVE bit switched
    int i0 = _idx(layer, row, col, board_shape);
    int16_t old_cell = board[i0];
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


static int check_for_violation(int16_t src, int16_t dst, int neighbors) {
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

static char print_cell(int16_t cell) {
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
}


static swap_cells_t swap_cells(
        int16_t *board, int *neighbors, int *violations, int *oscillations,
        board_shape_t board_shape, int row, int col, int16_t new_cell,
        iset *bad_idx) {

    swap_cells_t delta_swap = {0, 0};
    bounds_t area_of_effect = {col, row, col, row};
    int layer_size = board_shape.rows * board_shape.cols;
    int start_cell = board[_idx(0, row, col, board_shape)]; // debug only!

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
                int16_t b1 = board[i1], b2;
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
            int16_t b1 = board[i1], b2;
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
            delta_swap.violations += _violations - violations[i1];
            delta_swap.oscillations += (_oscillations == is_osc);
            delta_swap.oscillations -= (oscillations[i1] == is_osc);
            violations[i1] = _violations;
            oscillations[i1] = _oscillations;
            total_violations += _violations;
            if (bad_idx && _violations) {
                iset_add(bad_idx, i1);
            } else if (bad_idx) {
                iset_discard(bad_idx, i1);
            }
        }
    }
    return delta_swap;

    // Print out the area of effect
    PRINT("\nswap cells: %c -> %c\n", print_cell(start_cell), print_cell(new_cell));
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
    }

    return delta_swap;
}


int gen_oscillator(
        int16_t *board, int32_t *mask, int32_t *seeds, board_shape_t shape,
        double rel_max_iter, double rel_min_fill, double temperature,
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
    PRINT("\nBoard 1: ");
    for (int i=0; i < board_size; i++) PRINT("%i ", board[i]);

    PRINT("\nNeighbors 1: ");
    for (int i=0; i < board_size; i++) PRINT("%i ", neighbors[i]);

    for (int i=0; i < shape.depth; i++) {
        int temp[layer_size];
        wrapped_convolve(neighbors + i*layer_size, temp, shape.rows, shape.cols);
    }

    PRINT("\nNeighbors 2: ");
    for (int i=0; i < board_size; i++) PRINT("%i ", neighbors[i]);

    for (int i=0; i < layer_size; i++) {
        violations[i] = check_for_violation(
            board[i + last_layer_idx], board[i], neighbors[i + last_layer_idx]);
        if (violations[i])  iset_add(&bad_idx, i);
        if (seeds[i])  iset_add(&seeds_idx, i);
        if (mask[i]) {
            iset_add(&unmasked_idx, i);
            total_area++;
            int cell_type_idx = idx_for_cell_type2(board[i]);
            totals[cell_type_idx]++;
        }
    }

    PRINT("\n\n%i checkpoint...\n", __LINE__);

    // Calculate some constants for the loop
    int max_iter = rel_max_iter * total_area;
    max_iter *= shape.depth * shape.depth;
    int interior_area = calc_interior_area(mask, shape.rows, shape.cols);
    double effective_area = 0.75 * interior_area + 0.25 * total_area;
    double min_fill = rel_min_fill * effective_area;
    PRINT("Total area: %i; ", total_area);
    PRINT("Interior area: %i\n", interior_area);
    if (interior_area < 2) {
        PRINT("AREA TOO SMALL!");
        return AREA_TOO_SMALL_ERROR;
    }

    PRINT("\nSTARTING THE LOOP\n\n");

    // And start the loop!
    int num_iter;
    for (num_iter=0; num_iter < max_iter; num_iter++) {
        int not_empty = total_area - totals[EMPTY_IDX];
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
        double osc_bonus = 0.2;
        int neighborhood_size = (2*shape.depth+1) * (2*shape.depth+1);
        double log_probs[4 * neighborhood_size];
        int16_t cell_types[4 * neighborhood_size];
        int switched_idx[4 * neighborhood_size];
        double max_log_prob = -1e100;

        // Try switching each cell in the target's extended neighborhood.
        int num_switched = 0;
        for (int r = r0 - shape.depth; r <= r0 + shape.depth; r++) {
            for (int c = c0 - shape.depth; c <= c0 + shape.depth; c++) {
                //PRINT("\nr,c = %i,%i\n", r, c);
                int i1 = _idx(0, r, c, shape);
                if (!mask[i1]) continue;
                int16_t current_cell = board[i1];
                int start_idx = idx_for_cell_type2(current_cell) + 1;
                int delta_violations = 0;
                int delta_oscillations = 0;
                for (int j = start_idx; j < start_idx+3; j++) {
                    int16_t target_type = cell_type_array[j & 3];
                    swap_cells_t delta = swap_cells(
                        board, neighbors, violations, oscillations,
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
                    board, neighbors, violations, oscillations,
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
        double target_prob = rand() * total_prob / RAND_MAX;
        int k;
        for (k=0; k < num_switched; k++) {
            if (cum_probs[k] > target_prob) {
                // Pick this cell!
                int cell_idx = switched_idx[k];
                int16_t old_cell = board[cell_idx];
                int16_t new_cell = cell_types[k];
                PRINT("PICK: %c -> %c; i=(%i,%i); m=%i\n",
                    print_cell(old_cell), print_cell(new_cell),
                    cell_idx / shape.cols, cell_idx % shape.cols, mask[cell_idx]);
                swap_cells(
                    board, neighbors, violations, oscillations,
                    shape, cell_idx / shape.cols, cell_idx % shape.cols,
                    new_cell, &bad_idx);
                totals[idx_for_cell_type2(old_cell)]--;
                totals[idx_for_cell_type2(new_cell)]++;
                break;
            }
        }
        if (k >= num_switched) {
            PRINT("no cell picked... %g %g %g\n", target_prob, total_prob, max_log_prob);
        }
    }

    PRINT("%i checkpoint...\n", __LINE__);

    iset_free(&bad_idx);
    iset_free(&unmasked_idx);
    iset_free(&seeds_idx);
    PRINT("Iterations: %i/%i\n", num_iter, max_iter);
    PRINT("Num alive: %i/%i\n", totals[ALIVE_IDX], total_area);
    return 0;  // DEBUG!
    return num_iter == max_iter ? MAX_ITER_ERROR : 0;
}

