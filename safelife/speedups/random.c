#include <stdlib.h>
#include "random.h"

void random_seed(uint64_t seed) {
    srand(seed);
}

int32_t random_int(int32_t high) {
    return rand() % high;
}

float random_float() {
    return rand() / RAND_MAX;
}
