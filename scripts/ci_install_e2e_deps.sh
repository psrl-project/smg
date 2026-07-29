#!/bin/bash
# Install e2e test dependencies
# Usage: ci_install_e2e_deps.sh [extra_deps...]

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Installing e2e test dependencies..."
python3 -m pip install e2e_test/

# Install SmgClient (pure Python client for cross-SDK parity testing)
echo "Installing smg-client..."
python3 -m pip install clients/python/

# Install any extra dependencies passed as arguments
if [ $# -gt 0 ]; then
    echo "Installing extra dependencies: $@"
    python3 -m pip --no-cache-dir install --upgrade "$@"
fi

# Ensure protobuf 7 runtime so grpcio-reflection/health 1.82+/1.83 (pulled by
# smg-grpc-servicer / engine extras) can load their protobuf-7 gencode. This
# runs last in case earlier installs left an older runtime.
python3 -m pip install "protobuf>=7.35.1,<8"

echo "E2E test dependencies installed"
