#!/usr/bin/env bash
# bench.sh NAME [extra nemo-speech flags...]  -> one summary line in bench.tsv
set -uo pipefail
M=~/models/nemotron-3.5-transcribe
B=${NEMO_SPEECH_BIN:-$(dirname "$0")/../../build/metal-server/bin/nemo-speech}
D=${BENCH_DIR:-bench}
AUDIO=${AUDIO:-$M/samples/dialog.wav}
mkdir -p "$D"
name=$1; shift
base=(--model $M/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf --asr.salm.max_sequences 1
      --asr.salm.n_ctx_per_sequence 8192 --asr.salm.max_context 8192 --asr.salm.mtp_draft 0
      --asr.salm.encoder_max_batch 1 --no-warmup -f json --force)
swap0=$(sysctl -n vm.swapusage | awk '{print $6}')
NEMO_SALM_PROFILE=1 /usr/bin/time -l "$B" transcribe "$AUDIO" "${base[@]}" "$@" -o "$D/$name.json" > /dev/null 2> "$D/$name.log"
rc=$?
real=$(awk '/ real /{print $1}' "$D/$name.log")
peak=$(awk '/peak memory footprint/{printf "%.1f", $1/1e9}' "$D/$name.log")
prof=$(grep '^\[prof\]' "$D/$name.log" | awk '{printf "%s=%s ", $2, $3}')
swap1=$(sysctl -n vm.swapusage | awk '{print $6}')
diff=$(python3 - "$D/$name.json" "${REFERENCE:-$M/samples/out_dialog.json}" <<'EOF'
import json,sys,difflib,re
try:
    a=json.load(open(sys.argv[1]))['text']; b=json.load(open(sys.argv[2]))['text']
    norm=lambda t: re.sub(r"[^a-z0-9' ]"," ",t.lower()).split()
    sm=difflib.SequenceMatcher(None,norm(a),norm(b)); print(f"wordmatch={sm.ratio():.3f}")
except Exception as e: print("noout")
EOF
)
printf "%s\trc=%s\treal=%s\tpeakGB=%s\tswap=%s->%s\t%s\t%s\t%s\n" "$name" "$rc" "$real" "$peak" "$swap0" "$swap1" "$prof" "$diff" "$*" | tee -a "$D/bench.tsv"
