#include <error.h>
#include <misc.h>
namespace amgx {
namespace template_plugin { extern AMGX_ERROR initialize(); extern void finalize(); }
namespace eigensolvers { extern AMGX_ERROR initialize(); extern void finalize(); }

AMGX_ERROR initializePlugins() {
    if (AMGX_OK != template_plugin::initialize()) return AMGX_ERR_PLUGIN;
    if (AMGX_OK != eigensolvers::initialize()) return AMGX_ERR_PLUGIN;
  return AMGX_OK;
}

void finalizePlugins() {
  template_plugin::finalize();
  eigensolvers::finalize();
}
} // namespace amgx
