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


static PyObject *gen_still_life_py(PyObject *self, PyObject *args) {
    PyObject *board_obj, *mask_obj;
    PyArrayObject *board = NULL, *mask = NULL;

    if (!PyArg_ParseTuple(args, "OO", &board_obj, &mask_obj)) return NULL;
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
        PyArray_DIM(board, 1)
    );
    if (err_code && 0) {
        PY_RUN_ERROR("Max-iter hit. Aborting!");
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
        "gen_still_life", (PyCFunction)gen_still_life_py, METH_VARARGS,
        "Generate a still life pattern."
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
