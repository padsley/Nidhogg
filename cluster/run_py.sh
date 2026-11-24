#!/bin/bash
# Run a Python script with the dragon_nn venv. Condor resolves the venv's
# python symlink to /usr/bin/python3.12 when it is the job executable, which
# loses the venv's site-packages; going through this script avoids that.
source /data/pagroup/pa02/padsley/dragon_nn/cluster/env.sh
exec "$NIDHOGG_PY" "$@"
