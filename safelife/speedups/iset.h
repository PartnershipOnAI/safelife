typedef struct {
    // 'set' contains an unordered list of integers of length 'max_size'.
    // The first 'size' number of items belong to the set, the rest do not.
    // New integers less than 'max_size' can be added to the set by a simple
    // swap operation.
    int *set;
    int *ptr; // This is the inverse of set: set[ptr[k]] == k
    int size;
    int max_size;
} iset;


iset iset_alloc(int max_size);
void iset_free(iset *s);
void iset_add(iset *s, int val);
void iset_discard(iset *s, int val);
int iset_contains(iset *s, int val);
int iset_sample(iset *s);
