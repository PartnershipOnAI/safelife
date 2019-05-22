#include <stdint.h>


enum cell_type_bits {
    ALIVE_BIT = 0,
    MOVABLE_BIT = 2,
    DESTRUCTIBLE_BIT = 3,
    FROZEN_BIT = 4,
    PRESERVING_BIT = 5,
    INHIBITING_BIT = 6,
    SPAWNING_BIT = 7,
    COLOR_BIT = 9,

};

enum cell_types {
    ALIVE = 1 << ALIVE_BIT,
    MOVABLE = 1 << MOVABLE_BIT,
    DESTRUCTIBLE = 1 << DESTRUCTIBLE_BIT,
    FROZEN = 1 << FROZEN_BIT,
    PRESERVING = 1 << PRESERVING_BIT,
    INHIBITING = 1 << INHIBITING_BIT,
    SPAWNING = 1 << SPAWNING_BIT,
    COLORS = 7 << COLOR_BIT,
};

