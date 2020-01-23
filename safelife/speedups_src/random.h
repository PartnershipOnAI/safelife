#include <stdint.h>
#include <Python.h>

int set_bit_generator(PyObject *bit_generator);
int random_seed(uint32_t seed);
uint32_t random_int(uint32_t high);
double random_float(void);
