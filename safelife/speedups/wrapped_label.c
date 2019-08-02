#include "wrapped_label.h"
#include "iset.h"


int wrapped_label(int32_t *data, int nrow, int ncol) {
    int size = nrow * ncol;
    int current_label = 0;
    iset visited = iset_alloc(size);
    iset boundary = iset_alloc(size);


    while (visited.size < size) {
        // Get the next unvisited non-empty cell.
        int i0 = visited.set[visited.size++];
        if (!data[i0]) {
            continue;
        }

        // Clear the boundary set and add the new index to it.
        data[i0] = ++current_label;
        boundary.size = 0;
        iset_add(&boundary, i0);

        // Start expanding the boundary outwards, adding labels as we go.
        while (boundary.size > 0) {
            // Pop the last index on the boundary
            int i1 = boundary.set[--boundary.size];
            int r, c;
            r = i1 / ncol;
            c = i1 % ncol;

            for (int dy=-1; dy<=1; dy++) {
                int y = r + dy;
                if (y < 0) y += nrow;
                if (y >= nrow) y -= nrow;
                for (int dx=-1; dx<=1; dx++) {
                    int x = c + dx;
                    if (x < 0) x += ncol;
                    if (x >= ncol) x -= ncol;
                    int i2 = x + y * ncol;
                    if (data[i2] > 0 && !iset_contains(&visited, i2)) {
                        data[i2] = current_label;
                        iset_add(&boundary, i2);
                    }
                    iset_add(&visited, i2);
                }
            }
        }
    }

    iset_free(&visited);
    iset_free(&boundary);

    return current_label;
}
