#include <Python.h>
#include <numpy/arrayobject.h>
#include "advance_board.h"


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


static PyMethodDef test_methods[] = {
    {
      "advance_board", (PyCFunction)advance_board_py, METH_VARARGS,
      "Advances the board one step."
    },
    {NULL, NULL, 0, NULL}  /* Sentinel */
};


static struct PyModuleDef ext_module = {
   PyModuleDef_HEAD_INIT,
   "_ext",   /* name of module */
   NULL,     /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   test_methods
};


PyMODINIT_FUNC PyInit__ext(void) {
    import_array();
    return PyModule_Create(&ext_module);
}
