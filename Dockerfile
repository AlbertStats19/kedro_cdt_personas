# ================================================================
# 🧩 BASE IMAGE
# ================================================================
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} as runtime-environment

# ------------------------------------------------
# 🔧 Actualiza pip y prepara uv (instalador rápido)
# ------------------------------------------------
RUN python -m pip install -U "pip>=21.2"
RUN pip install uv

# ------------------------------------------------
# 📦 Instala dependencias del proyecto Kedro
# ------------------------------------------------
COPY requirements.txt /tmp/requirements.txt

# Instala librerías necesarias para Kedro y ML comunes
RUN uv pip install --system --no-cache-dir -r /tmp/requirements.txt && \
    uv pip install --system --no-cache-dir scikit-learn==1.4.0 && \
    uv pip install --system --no-cache-dir kedro==0.18.14 && \
    rm -f /tmp/requirements.txt

# ------------------------------------------------
# 👤 Crea usuario seguro (no root)
# ------------------------------------------------
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

# ================================================================
# 🧩 BUILD FINAL IMAGE
# ================================================================
FROM runtime-environment

# Copia el proyecto completo (respetando .dockerignore)
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

# Exponer puerto (solo si quieres debug o Jupyter)
EXPOSE 8888

# ------------------------------------------------
# 🚀 Comando por defecto (se puede sobreescribir)
# ------------------------------------------------
CMD ["kedro", "run"]
