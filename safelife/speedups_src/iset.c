#include <stdlib.h>
#include "iset.h"
#include "random.h"


iset iset_alloc(int max_size) {
    iset s = {malloc(sizeof(int) * 2*max_size), NULL, 0, max_size};
    s.ptr = s.set + max_size;
    for (int k = 0; k < max_size; k++) {
        s.set[k] = k;
        s.ptr[k] = k;
    }
    return s;
}

void iset_free(iset *s) {
    free(s->set);
    s->max_size = 0;
    s->size = 0;
}

void iset_add(iset *s, int val) {
    // assert val < s->max_size
    int k = s->ptr[val];
    int y;
    if (k >= s->size) {
        // swap the index at k and at the top of the set
        // move the size index up one
        y = s->set[s->size];
        s->set[s->size] = val;
        s->ptr[val] = s->size;
        s->set[k] = y;
        s->ptr[y] = k;
        s->size++;
    }
}

void iset_discard(iset *s, int val) {
    // assert val < s->max_size
    int k = s->ptr[val];
    int y;
    if (k < s->size) {
        // move the size index down one
        // swap the index at k and at the top of the set
        s->size--;
        y = s->set[s->size];
        s->set[s->size] = val;
        s->ptr[val] = s->size;
        s->set[k] = y;
        s->ptr[y] = k;
    }
}

int iset_contains(iset *s, int val) {
    return s->ptr[val] < s->size;
}

int iset_sample(iset *s) {
    // If the size is zero, sample from the whole array.
    int k = random_int(s->size ? s->size : s->max_size);
    return s->set[k];
}
