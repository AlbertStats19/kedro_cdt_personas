"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html.
"""

# ============================================================================
# 📦 PROJECT HOOKS
# ============================================================================
from data_bbog_integration_fabrica_personas.hooks import MemoryProfilingHooks
HOOKS = (MemoryProfilingHooks(),)

# ============================================================================
# ⚙️ CONFIG LOADER
# ============================================================================
from kedro.config import OmegaConfigLoader
from datetime import datetime
from dateutil.relativedelta import relativedelta


# ✅ Función mejorada: acepta '202507' o '2025-07-10'
def get_previous_month(period: str):
    if len(period) == 6:
        date = datetime.strptime(period, "%Y%m")
    else:
        date = datetime.strptime(period, "%Y-%m-%d")
    previous = date - relativedelta(months=1)
    return previous.strftime("%Y%m")


def get_execution_date(fecha_ejecucion: str):
    date = datetime.strptime(fecha_ejecucion, "%Y-%m-%d")
    return date.strftime("%Y%m")


def get_current_month():
    date = datetime.now()
    return date.strftime("%Y%m")


CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "custom_resolvers": {
        "previous_month": get_previous_month,
        "current_month": get_current_month,
        "format_execution_date": get_execution_date,
    }
}

# ============================================================================
# (Opcional) otros componentes de Kedro
# ============================================================================
# SESSION_STORE_CLASS = BaseSessionStore
# SESSION_STORE_ARGS = {"path": "./sessions"}
# CONF_SOURCE = "conf"
# CONTEXT_CLASS = KedroContext
# DATA_CATALOG_CLASS = DataCatalog
