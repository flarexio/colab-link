# colab-link service: NATS endpoints + JetStream job queue + warm Colab workers.
#
#   docker build -t flarexio/colab-link .      (released images: amd64 + arm64)
#   docker run -d --name colab-link \
#     -e COLAB_LINK_DRIVE_PROJECT=<gcp-project> \
#     -v ~/.flarex/colab-link:/root/.flarex/colab-link:ro \
#     -v ~/.config/colab-cli:/root/.config/colab-cli \
#     -v colab-link-state:/data \
#     flarexio/colab-link
#
# The colab CLI (Python) is the control plane for Colab VMs, so the service stays
# Python; the CLI is installed into the same venv as colab-link.
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server --no-install-project
COPY colab_link ./colab_link
COPY tools ./tools
COPY jobs ./jobs
COPY profiles.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server \
 # pyarrow (~150 MB) comes in through jupyter-kernel-client; the colab CLI never imports it.
 && uv pip uninstall -q --python /app/.venv/bin/python pyarrow

FROM python:3.13-slim
COPY --from=build /app /app
WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    COLAB_CLI_PYTHON=/app/.venv/bin/python \
    COLAB_LINK_STATE=/data/state
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["colab-link"]
CMD ["serve"]
