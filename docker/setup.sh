#!/usr/bin/env bash

# bash strict mode
set -euo pipefail

echo "Setting up SWE-bench Docker image..."
docker build -t opendevin/swe-bench:latest -f docker/Dockerfile . --build-arg TARGETARCH=$(uname -m)

echo "Done with setup!"
