/*
    This file lets us make use of numpy random number generation from
    C extensions. The state can be seeded or a generator function can be
    set directly.
*/

#define NO_IMPORT_ARRAY

#include <numpy/arrayobject.h>
#include <numpy/random/bitgen.h>
#include <numpy/random/distributions.h>
#include <stdlib.h>
#include "random.h"


// Save a global bit generator and bit generator state.
// Hanging onto a reference to the python object part is (probably) necessary
// in order to make sure that the state doesn't get freed during garbage
// collection.

static bitgen_t *bitgen_state = NULL;
static PyObject *bit_generator = NULL;


int set_bit_generator(PyObject *bitgen) {
    PyObject *capsule = NULL;

    if (!(capsule = PyObject_GetAttrString(bitgen, "capsule"))) goto error;
    if (!(bitgen_state = PyCapsule_GetPointer(capsule, "BitGenerator"))) goto error;

    bit_generator = bitgen;
    Py_INCREF(bit_generator);
    Py_XDECREF(capsule);
    return 1;

    error:
    Py_XDECREF(capsule);
    return 0;
}

int random_seed(uint32_t seed) {
    PyObject *np_random = NULL,
             *gen_func = NULL,
             *generator = NULL,
             *bitgen = NULL,
             *capsule = NULL;
    if (!(np_random = PyImport_ImportModule("numpy.random"))) goto error;
    if (!(gen_func = PyObject_GetAttrString(np_random, "default_rng"))) goto error;
    if (seed > 0) {
        if (!(generator = PyObject_CallFunction(gen_func, "I", seed))) goto error;
    }
    else {
        if (!(generator = PyObject_CallObject(gen_func, NULL))) goto error;
    }
    if (!(bitgen = PyObject_GetAttrString(generator, "bit_generator"))) goto error;
    if (!(capsule = PyObject_GetAttrString(bitgen, "capsule"))) goto error;
    if (!(bitgen_state = PyCapsule_GetPointer(capsule, "BitGenerator"))) goto error;

    Py_XDECREF(np_random);
    Py_XDECREF(gen_func);
    Py_XDECREF(generator);
    bit_generator = bitgen;
    Py_XDECREF(capsule);
    return 1;

    error:
    Py_XDECREF(np_random);
    Py_XDECREF(gen_func);
    Py_XDECREF(generator);
    Py_XDECREF(bitgen);
    Py_XDECREF(capsule);
    return 0;
}


double random_float(void) {
    if (!bitgen_state) {
        if (!random_seed(0)) {
            return 0.0;
        }
    }
    return bitgen_state->next_double(bitgen_state->state);
}


uint32_t random_int(uint32_t high) {
    // This is more-or-less copied from numpy/random/distributions.c,
    // except that the top value is exclusive rather than inclusive.
    // For some reason I'm having a hard time linking to the functions in
    // distributions.h directly.
    uint32_t mask, value;

    if (!bitgen_state) {
        if (!random_seed(0)) {
            return 0;
        }
    }
    if (high == 0) {
        return 0;
    }

    mask = high;

    /* Smallest bit mask >= high */
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;

    do {
        value = mask & bitgen_state->next_uint32(bitgen_state->state);
    } while (value >= high);

    return value;
}
