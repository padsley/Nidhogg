#!/bin/bash
# One Ancalagon run per condor job. Arguments (one jobs.txt line):
#   NAME MODE INPUT NEVENTS SEED PRESSURE_TORR GASFILE EXTRA
#   MODE     reaction | library
#   INPUT    reaction: .reaction file;  library: LIBRARY_CELLS file
#   NEVENTS  reaction: events;          library: events per cell
#   GASFILE  library: reaction file whose TARG card picks H2/He; reaction: "-"
#   EXTRA    "-" or comma-separated KEY=VALUE env settings
# Output goes to $DRAGON_NN/runs/<batch>/<NAME>/ (NAME may contain one "/"):
#   out.root (renamed from out.root.partial only on success), run.log,
#   status.txt (host, cpu, start/end, wall seconds, exit code).
# A job whose status.txt already says exit=0 is skipped, so a batch can be
# resubmitted as a whole after failures.
set -u
NAME=$1 MODE=$2 INPUT=$3 NEVENTS=$4 SEED=$5 PRESSURE=$6 GASFILE=$7 EXTRA=$8
source /data/pagroup/pa02/padsley/dragon_nn/cluster/env.sh

DIR=$DRAGON_NN/runs/$NAME
mkdir -p "$DIR"
if grep -qs "^exit=0$" "$DIR/status.txt"; then
  echo "already done: $DIR"; exit 0
fi
cd "$DIR"
export TMPDIR=$DIR
rm -f out.root out.root.partial

export OUTPUT_MODE=compact OUTPUT_FILE=out.root.partial TRAJ_QUIET=1 EVENT_PRINT=0
export RANDOM_SEED=$SEED GAS_PRESSURE_TORR=$PRESSURE GAS_TEMPERATURE_K=300
if [ "$EXTRA" != "-" ]; then
  IFS=',' read -ra KV <<< "$EXTRA"
  for kv in "${KV[@]}"; do export "$kv"; done
fi

case $MODE in
  reaction) export REACTION_INPUT=$INPUT; ARGS=(--track-reaction "$NEVENTS") ;;
  library)  export REACTION_INPUT=$GASFILE LIBRARY_CELLS=$INPUT LIBRARY_EVENTS_PER_CELL=$NEVENTS
            ARGS=(--gamma-library) ;;
  *) echo "bad MODE $MODE" >&2; exit 2 ;;
esac

T0=$(date +%s)
{ echo "host=$(hostname)"; echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
  echo "start=$(date -Is)"; echo "mode=$MODE"; echo "nevents_arg=$NEVENTS"; echo "seed=$SEED"
  echo "pressure=$PRESSURE"; echo "input=$INPUT"; } > status.txt.partial
/usr/bin/time -f "maxrss_kb=%M" -o time.txt "$ANCALAGON_BIN" "${ARGS[@]}" > run.log 2>&1
RC=$?
T1=$(date +%s)
if [ $RC -eq 0 ]; then
  if [ -s out.root.partial ]; then mv out.root.partial out.root; else RC=3; fi
fi
{ cat status.txt.partial; cat time.txt 2>/dev/null; echo "end=$(date -Is)"; echo "wall_s=$((T1-T0))"
  echo "exit=$RC"; } > status.txt
rm -f status.txt.partial time.txt
exit $RC
