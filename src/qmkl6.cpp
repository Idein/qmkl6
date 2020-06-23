#include "qmkl6.h"
#include "qmkl6_internal.hpp"


qmkl6_context qmkl6;

qmkl6_context::qmkl6_context(void)
{
    this->init_support();

    this->unif = (uint32_t*) this->alloc_memory(sizeof(uint32_t) * 1024,
            this->unif_handle, this->unif_bus);
}

qmkl6_context::~qmkl6_context(void)
{
    this->finalize_support();
}
