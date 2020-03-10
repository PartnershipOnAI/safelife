#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "advance_board.h"
#include "gen_board.h"
#include "wrapped_label.h"
#include "random.h"
#include "fast_render.h"

#define PY_RUN_ERROR(msg) {PyErr_SetString(PyExc_RuntimeError, msg); goto error;}
#define PY_VAL_ERROR(msg) {PyErr_SetString(PyExc_ValueError, msg); goto error;}


static PyObject *BoardGenException;
static PyObject *MaxIterException;
static PyObject *BadProbabilityException;
static PyObject *InsufficientAreaException;


static PyObject *advance_board_py(PyObject *self, PyObject *args) {
    PyObject *board_obj;
    PyArrayObject *b1, *b2;
    float spawn_prob = 0.3;

    if (!PyArg_ParseTuple(args, "O|f", &board_obj, &spawn_prob)) return NULL;
    board_obj = PyArray_FROM_OTF(
        board_obj, NPY_UINT16, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!board_obj)  return NULL;
    b1 = (PyArrayObject *)board_obj;
    if (PyArray_NDIM(b1) != 2 || PyArray_SIZE(b1) == 0) {
        Py_DECREF(board_obj);
        return NULL;
    }
    b2 = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_UINT16, NPY_ARRAY_ENSURECOPY);
    Py_BEGIN_ALLOW_THREADS
    advance_board(
        (uint16_t *)PyArray_DATA(b1),
        (uint16_t *)PyArray_DATA(b2),
        PyArray_DIM(b1, 0),
        PyArray_DIM(b1, 1),
        spawn_prob
    );
    Py_END_ALLOW_THREADS
    Py_DECREF(board_obj);
    return (PyObject *)b2;
}


static char wrapped_label_doc[] =
    "wrapped_label(data)\n--\n\n"
    "Similar to :func:`ndimage.label`, but uses wrapped boundary conditions.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "data : ndarray\n"
    "    Data from which to select labels. Must be two-dimensional.\n"
    "    Non-zero elements are treated as features, while zero elements are\n"
    "    treated as background.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "labels : ndarray\n"
    "    An integer array of labels. Same shape as input.\n"
    "num_features : int\n";


static PyObject *wrapped_label_py(PyObject *self, PyObject *args) {
    PyObject *data;
    PyArrayObject *arr;

    if (!PyArg_ParseTuple(args, "O", &data)) return NULL;
    data = PyArray_FROM_OTF(
        data, NPY_INT32,
        NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST | NPY_ARRAY_ENSURECOPY);
    if (!data)  return NULL;
    arr = (PyArrayObject *)data;
    if (PyArray_NDIM(arr) != 2 || PyArray_SIZE(arr) == 0) {
        Py_DECREF(data);
        return NULL;
    }
    int num_labels;
    Py_BEGIN_ALLOW_THREADS
    num_labels = wrapped_label(
        (int32_t *)PyArray_DATA(arr),
        PyArray_DIM(arr, 0),
        PyArray_DIM(arr, 1)
    );
    Py_END_ALLOW_THREADS

    PyObject *rval = Py_BuildValue("Oi", data, num_labels);
    Py_DECREF(data);
    return rval;
}


static char gen_pattern_doc[] =
    "gen_pattern(board, mask, period, max_iter=40, min_fill=0.2, temperature=0.5, "
        "alive=(0,0), wall=(100,100), tree=(100,100))\n"
    "--\n\n"
    "Generate a random (potentially oscillating) pattern.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "board : ndarray\n"
    "mask : ndarray\n"
    "period: int\n"
    "    Oscillation period of the pattern. If 1, the pattern will be still.\n"
    "    Note that the algorithmic complexity increases very quickly with\n"
    "    increasing period. Periods larger than 2 can take a long time.\n"
    "seeds : ndarray\n"
    "    Locations at which to start growing patterns. Same shape as board.\n"
    "max_iter : float\n"
    "    Maximum number of iterations to be run, relative to the board size.\n"
    "min_fill : float\n"
    "    Minimum fraction of the (unmasked) board that must be populated.\n"
    "temperature : float\n"
    "osc_bonus : float\n"
    "    Bonus applied to cells that are oscillating. Defaults to 0.3.\n"
    "    Larger bonuses encourage oscillators, as opposed to still lifes, but\n"
    "    may also make the algorithm slow to converge.\n"
    "alive : (float, float)\n"
    "    Penalties for 'life' cells.\n"
    "    First value is penalty at zero density, second is the penalty when\n"
    "    life cells take up 100% of the populated area. Penalty increase is\n"
    "    linear.\n"
    "wall : (float, float)\n"
    "    Penalties for 'wall' cells. Defaults to (100, 100), basically\n"
    "    guaranteeing that walls don't form.\n"
    "tree : (float, float)\n"
    "    Penalties for 'tree' cells. Defaults to (100, 100), basically\n"
    "    guaranteeing that trees don't form.\n"
;

static PyObject *gen_pattern_py(PyObject *self, PyObject *args, PyObject *kw) {
    PyObject *board_obj, *mask_obj, *seeds_obj = Py_None;
    PyArrayObject *board = NULL, *mask = NULL, *seeds = NULL;

    int period = 1;
    double max_iter = 40;
    double min_fill = 0.2;
    double temperature = 0.5;
    double osc_bonus = 0.3;
    double cp[8] = {
        // intercept and slope of penalty
        0, 0,  // EMPTY (handled separately by min_fill)
        100, 100,  // WALL
        0, 0,  // ALIVE
        100, 100,  // TREE
    };
    static char *kwlist[] = {
        "board", "mask", "period", "seeds", "max_iter", "min_fill",
        "temperature", "osc_bonus", "alive", "wall", "tree",
        NULL
    };
    uint16_t *layers = NULL;

    if (!PyArg_ParseTupleAndKeywords(
            args, kw, "OOi|Odddd(dd)(dd)(dd):gen_still_life",
            kwlist,
            &board_obj, &mask_obj, &period, &seeds_obj,
            &max_iter, &min_fill, &temperature, &osc_bonus,
            cp+4, cp+5, cp+2, cp+3, cp+6, cp+7)) {
        return NULL;
    }

    for (int i = 0; i < 4; i++) {
        // Convert penalties to intercept, slope instead of t=0, t=1
        cp[2*i+1] -= cp[2*i];
    }

    board = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_UINT16,
        NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
    mask = (PyArrayObject *)PyArray_FROM_OTF(
        mask_obj, NPY_INT32,
        NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
    // Make seeds same as mask if not available
    seeds = seeds_obj != Py_None ? (PyArrayObject *)PyArray_FROM_OTF(
        seeds_obj, NPY_INT32,
        NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSURECOPY) : mask;
    if (!board || !mask || !seeds) {
        PY_VAL_ERROR("Board and/or mask and/or seeds are not arrays.");
    }
    if (period <= 0) {
        PY_VAL_ERROR("Pattern period must be larger than 0.");
    }
    if (PyArray_NDIM(board) != 2 || PyArray_NDIM(mask) != 2 || PyArray_NDIM(seeds) != 2) {
        PY_VAL_ERROR("Board, mask, and seeds must all be dimension 2.");
    }
    if (PyArray_DIM(board, 0) < 3 || PyArray_DIM(board, 1) < 3) {
        PY_VAL_ERROR("Board must be at least 3x3.");
    }
    if (PyArray_DIM(board, 0) != PyArray_DIM(mask, 0) ||
            PyArray_DIM(board, 1) != PyArray_DIM(mask, 1) ||
            PyArray_DIM(board, 0) != PyArray_DIM(seeds, 0) ||
            PyArray_DIM(board, 1) != PyArray_DIM(seeds, 1)) {
        PY_VAL_ERROR("Board, mask, and seeds must all have the same shape.");
    }

    // Create a new temporary board array
    board_shape_t board_shape = {
        period, PyArray_DIM(board, 0), PyArray_DIM(board, 1)
    };
    int layer_size = board_shape.cols * board_shape.rows;
    int board_size = board_shape.depth * layer_size;
    int err_code;

    layers = malloc(sizeof(uint16_t) * board_size);
    if (!layers)  {
        PyErr_NoMemory();
        goto error;
    }
    memcpy(layers, PyArray_DATA(board), sizeof(uint16_t) * layer_size);

    // Advance to the next timestep
    Py_BEGIN_ALLOW_THREADS

    for (int n = 1; n < board_shape.depth; n++) {
        advance_board(
            layers + (n-1)*layer_size, layers + n*layer_size,
            board_shape.rows, board_shape.cols, 0.0);
    }

    err_code = gen_pattern(
        layers,
        (int32_t *)PyArray_DATA(mask),
        (int32_t *)PyArray_DATA(seeds),
        board_shape, max_iter, min_fill, temperature, osc_bonus, cp
    );

    Py_END_ALLOW_THREADS

    switch (err_code) {
        case 0:
            memcpy(PyArray_DATA(board), layers, sizeof(uint16_t) * layer_size);
            goto success;
        case MAX_ITER_ERROR:
            PyErr_SetString(MaxIterException, "Max-iter hit. Aborting!");
            goto error;
        case AREA_TOO_SMALL_ERROR:
            PyErr_SetString(InsufficientAreaException,
                "The unmasked area was too small to generate a pattern.");
            goto error;
        default:
            PyErr_SetString(BoardGenException, "Miscellany error.");
            goto error;
    }

    success:
    Py_DECREF(mask);
    if (seeds_obj != Py_None) Py_DECREF(seeds);
    free(layers);
    return (PyObject *)board;

    error:
    Py_XDECREF((PyObject *)board);
    Py_XDECREF((PyObject *)mask);
    if (seeds_obj != Py_None) Py_XDECREF(seeds);
    if (layers) free(layers);
    return NULL;
}


static PyObject *seed_py(PyObject *self, PyObject *args) {
    unsigned int i;
    if (!PyArg_ParseTuple(args, "I", &i)) return NULL;
    if(!random_seed(i)) return NULL;
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *set_bit_generator_py(PyObject *self, PyObject *args) {
    PyObject *bit_gen_obj;
    if (!PyArg_ParseTuple(args, "O", &bit_gen_obj)) return NULL;
    if (!set_bit_generator(bit_gen_obj)) return NULL;
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *render_board_py(PyObject *self, PyObject *args) {
    PyObject *board_obj, *goals_obj, *orientation_obj, *sprites_obj;
    PyArrayObject
        *board = NULL,
        *goals = NULL,
        *orientation = NULL,
        *sprites = NULL,
        *out = NULL;

    if (!PyArg_ParseTuple(
            args, "OOOO",
            &board_obj, &goals_obj, &orientation_obj, &sprites_obj)) {
        return NULL;
    }

    board = (PyArrayObject *)PyArray_FROM_OTF(
        board_obj, NPY_UINT16, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    goals = (PyArrayObject *)PyArray_FROM_OTF(
        goals_obj, NPY_UINT16, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    orientation = (PyArrayObject *)PyArray_FROM_OTF(
        orientation_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    sprites = (PyArrayObject *)PyArray_FROM_OTF(
        sprites_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);

    // bunch of error checking
    if (!board || !goals || !orientation || !sprites) {
        PY_VAL_ERROR("All inputs must be numpy arrays.");
    }
    int ndim = PyArray_NDIM(board);
    npy_intp *dims = PyArray_SHAPE(board);
    if (ndim < 2) {
        PY_VAL_ERROR("Board must have at least two dimensions.");
    }
    int depth = PyArray_SIZE(board) / (dims[ndim-1] * dims[ndim-2]);
    if (PyArray_SIZE(board) != PyArray_SIZE(goals)) {
        PY_VAL_ERROR("Board and goals must have same size.");
    }
    if (PyArray_SIZE(orientation) != depth) {
        PY_VAL_ERROR("Only one orientation allowed per board.");
    }
    if (PyArray_SIZE(sprites) != 70*70*4) {
        PY_VAL_ERROR("Sprites should have shape (70, 70, 4).");
    }

    // Create the output array
    npy_intp *out_dims = malloc(sizeof(npy_intp) * (ndim+1));
    for (int k=0; k<ndim-2; k++) {
        out_dims[k] = dims[k];
    }
    out_dims[ndim-2] = dims[ndim-2] * SPRITE_SIZE;
    out_dims[ndim-1] = dims[ndim-1] * SPRITE_SIZE;
    out_dims[ndim] = 3;
    out = (PyArrayObject *)PyArray_SimpleNew(ndim+1, out_dims, NPY_UINT8);
    free(out_dims);
    if (!out) {
        PY_VAL_ERROR("Could not allocate output array");
    }

    // render!
    Py_BEGIN_ALLOW_THREADS
    render_board(
        (uint16_t *)PyArray_DATA(board),
        (uint16_t *)PyArray_DATA(goals),
        (uint8_t *)PyArray_DATA(orientation),
        dims[ndim-1], dims[ndim-2], depth,
        (float *)PyArray_DATA(sprites),
        (uint8_t *)PyArray_DATA(out)
    );
    Py_END_ALLOW_THREADS

    Py_DECREF((PyObject *)board);
    Py_DECREF((PyObject *)goals);
    Py_DECREF((PyObject *)orientation);
    Py_DECREF((PyObject *)sprites);
    return (PyObject *)out;

    error:
    Py_XDECREF((PyObject *)board);
    Py_XDECREF((PyObject *)goals);
    Py_XDECREF((PyObject *)orientation);
    Py_XDECREF((PyObject *)sprites);
    Py_XDECREF((PyObject *)out);
    return NULL;
}


static PyMethodDef methods[] = {
    {
        "advance_board", (PyCFunction)advance_board_py, METH_VARARGS,
        "Advances the board one step."
    },
    {
        "gen_pattern", (PyCFunction)gen_pattern_py,
        METH_VARARGS | METH_KEYWORDS, gen_pattern_doc
    },
    {
        "wrapped_label", (PyCFunction)wrapped_label_py,
        METH_VARARGS | METH_KEYWORDS, wrapped_label_doc
    },
    {
        "_render_board", (PyCFunction)render_board_py,
        METH_VARARGS | METH_KEYWORDS, NULL
    },
    {
        "seed", (PyCFunction)seed_py, METH_VARARGS,
        "Seed the random number generator. (deprecated)"
    },
    {
        "set_bit_generator", (PyCFunction)set_bit_generator_py, METH_VARARGS,
        "Sets the bit generator for random functions.\n\n"
        "Parameters\n"
        "----------\n"
        "bit_generator : numpy.random.BitGenerator\n"
    },
    {NULL, NULL, 0, NULL}  /* Sentinel */
};


static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "safelife.speedups",   /* name of module */
    "C extensions for SafeLife to speed up game physics and procedural generation.",
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    methods
};

static PyObject *BoardGenException;
static PyObject *MaxIterException;
static PyObject *BadProbabilityException;
static PyObject *InsufficientAreaException;


PyMODINIT_FUNC PyInit_speedups(void) {
    import_array();

    PyObject *m = PyModule_Create(&module_def);
    if (!m) return NULL;

    BoardGenException = PyErr_NewException(
        "speedups.BoardGenException", NULL, NULL);
    MaxIterException = PyErr_NewException(
        "speedups.MaxIterException", BoardGenException, NULL);
    BadProbabilityException = PyErr_NewException(
        "speedups.BadProbabilityException", BoardGenException, NULL);
    InsufficientAreaException = PyErr_NewException(
        "speedups.InsufficientAreaException", BoardGenException, NULL);
    PyModule_AddObject(m, "BoardGenException", BoardGenException);
    PyModule_AddObject(m, "MaxIterException", MaxIterException);
    PyModule_AddObject(m, "BadProbabilityException", BadProbabilityException);
    PyModule_AddObject(m, "InsufficientAreaException", InsufficientAreaException);
    PyModule_AddObject(m, "NEW_CELL_MASK", PyLong_FromLong(1));
    PyModule_AddObject(m, "CAN_OSCILLATE_MASK", PyLong_FromLong(2));
    PyModule_AddObject(m, "INCLUDE_VIOLATIONS_MASK", PyLong_FromLong(4));

    return m;
}
