#include <Python.h>
#include <numpy/arrayobject.h>
#include "advance_board.h"
#include "gen_board.h"

#define PY_RUN_ERROR(msg) {PyErr_SetString(PyExc_RuntimeError, msg); goto error;}
#define PY_VAL_ERROR(msg) {PyErr_SetString(PyExc_ValueError, msg); goto error;}


static PyObject *advance_board_py(PyObject *self, PyObject *args) {
    PyObject *board_obj;
    PyArrayObject *b1, *b2;
    float spawn_prob;

    if (!PyArg_ParseTuple(args, "Of", &board_obj, &spawn_prob)) return NULL;
    board_obj = PyArray_FROM_OTF(
        board_obj, NPY_INT16, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!board_obj)  return NULL;
    b1 = (PyArrayObject *)board_obj;
    if (PyArray_NDIM(b1) != 2 || PyArray_SIZE(b1) == 0) {
        Py_DECREF(board_obj);
        return NULL;
    }
    b2 = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_INT16, NPY_ARRAY_ENSURECOPY);
    advance_board(
        (int16_t *)PyArray_DATA(b1),
        (int16_t *)PyArray_DATA(b2),
        PyArray_DIM(b1, 0),
        PyArray_DIM(b1, 1),
        spawn_prob
    );
    Py_DECREF(board_obj);
    return (PyObject *)b2;
}

static char gen_still_life_doc[] =
    "gen_still_life(board, mask, max_iter=40, min_fill=0.2, temperature=0.5, "
        "alive=(0,0), wall=(100,100), tree=(100,100), weed=(100,100), "
        "predator=(100,100), icecube=(100,100), fountain=(100,100))\n"
    "--\n\n"
    "Generate a still life pattern.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "board : ndarray\n"
    "mask : ndarray\n"
    "max_iter : float\n"
    "    Maximum number of iterations to be run, relative to the board size.\n"
    "min_fill : float\n"
    "    Minimum fraction of the (unmasked) board that must be populated.\n"
    "temperature : float\n"
    "alive : (float, float)\n"
    "    Penalties for 'life' cells.\n"
    "    First value is penalty at zero density, second is the penalty when\n"
    "    life cells take up 100% of the populated area. Penalty increase is\n"
    "    linear.\n"
    "wall : (float, float)\n"
    "    Penalties for 'wall' cells.\n"
    "tree : (float, float)\n"
    "    Penalties for 'tree' cells.\n"
    "weed : (float, float)\n"
    "    Penalties for 'weed' cells.\n"
    "predator : (float, float)\n"
    "    Penalties for 'predator' cells.\n"
    "ice_cube : (float, float)\n"
    "    Penalties for 'icecube' cells.\n"
    "fountain : (float, float)\n"
    "    Penalties for 'fountain' cells.\n"
;

static PyObject *gen_still_life_py(PyObject *self, PyObject *args, PyObject *kw) {
    PyObject *board_obj, *mask_obj;
    PyArrayObject *board = NULL, *mask = NULL;

    double max_iter = 40;
    double min_fill = 0.2;
    double temperature = 0.5;
    double cp[16] = {
        // intercept and slope of penalty
        0, 0,  // EMPTY (handled separately by min_fill)
        0, 0,  // ALIVE
        100, 100,  // WALL
        100, 100,  // TREE
        100, 100,  // WEED
        100, 100,  // PREDATOR
        100, 100,  // ICECUBE
        100, 100,  // FOUNTAIN
    };
    static char *kwlist[] = {
        "board", "mask", "max_iter", "min_fill", "temperature",
        "alive", "wall", "tree", "weed", "predator", "ice_cube", "fountain",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "OO|ddd(dd)(dd)(dd)(dd)(dd)(dd)(dd):gen_still_life",
            kwlist,
            &board_obj, &mask_obj, &max_iter, &min_fill, &temperature,
            cp+2, cp+3, cp+4, cp+5, cp+6, cp+7, cp+8, cp+9,
            cp+10, cp+11, cp+12, cp+13, cp+14, cp+15)) {
        return NULL;
    }

    for (int i = 0; i < 8; i++) {
        // Convert penalties to intercept, slope instead of t=0, t=1
        cp[2*i+1] -= cp[2*i];
    }

    board = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_INT16,
        NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
    mask = (PyArrayObject *)PyArray_FROM_OTF(
        mask_obj, NPY_INT32,
        NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY);
    if (!board || !mask) {
        PY_VAL_ERROR("Board and/or mask are not arrays.");
    }
    if (PyArray_NDIM(board) != 2 || PyArray_NDIM(mask) != 2) {
        PY_VAL_ERROR("Board and mask must be dimension 2.");
    }
    if (PyArray_DIM(board, 0) < 3 || PyArray_DIM(board, 1) < 3) {
        PY_VAL_ERROR("Board and mask must be at least 3x3.");
    }
    if (PyArray_DIM(board, 0) != PyArray_DIM(mask, 0) ||
            PyArray_DIM(board, 1) != PyArray_DIM(mask, 1)) {
        PY_VAL_ERROR("Board and mask must have the same dimensions.");
    }

    int err_code = gen_still_life(
        (int16_t *)PyArray_DATA(board),
        (int32_t *)PyArray_DATA(mask),
        PyArray_DIM(board, 0),
        PyArray_DIM(board, 1),
        max_iter, min_fill, temperature, cp
    );
    if (err_code) {
        if (err_code == MAX_ITER_ERROR) {
            PY_RUN_ERROR("Max-iter hit. Aborting!");
        } else {
            PY_RUN_ERROR("Miscellany error.")
        }
    }

    Py_DECREF(mask);
    return (PyObject *)board;

    error:
    Py_XDECREF((PyObject *)board);
    Py_XDECREF((PyObject *)mask);
    return NULL;
}


static PyObject *seed_py(PyObject *self, PyObject *args) {
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) return NULL;
    srand(i);
    Py_INCREF(Py_None);
    return Py_None;
}


static PyMethodDef methods[] = {
    {
        "advance_board", (PyCFunction)advance_board_py, METH_VARARGS,
        "Advances the board one step."
    },
    {
        "gen_still_life", (PyCFunction)gen_still_life_py,
        METH_VARARGS | METH_KEYWORDS, gen_still_life_doc
    },
    {
        "seed", (PyCFunction)seed_py, METH_VARARGS,
        "Seed the random number generator."
    },
    {NULL, NULL, 0, NULL}  /* Sentinel */
};


static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_ext",   /* name of module */
    NULL,     /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    methods
};


PyMODINIT_FUNC PyInit_speedups(void) {
    import_array();
    return PyModule_Create(&module_def);
}
