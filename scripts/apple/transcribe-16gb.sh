#!/usr/bin/env bash
# Nemotron 3.5 Transcribe (30B-A4B) on a 16 GB Apple-silicon Mac, with routed experts streamed from
# the SSD into a Metal expert cache (see "Nemotron 3.5 Transcribe on a 16 GB Mac" in README.md).
#
#   scripts/apple/transcribe-16gb.sh INPUT [extra nemo-speech transcribe flags...]
#     INPUT: any audio/video ffmpeg reads (converted to 16 kHz mono WAV first), or a directory of WAVs
#            (one model load for the whole directory).
#   examples:
#     transcribe-16gb.sh memo.m4a                              # plain text to stdout
#     transcribe-16gb.sh call.wav --diarize -f json --word-times -o call.json
#     transcribe-16gb.sh memo.wav -l de --translate-to en
#
# Environment:
#   NEMOTRON_MODEL  decoder GGUF (default: AtomicChat AD-Q2_0_M in ~/models/nemotron-3.5-transcribe);
#                   its Encoder-F16 GGUF must sit next to it
#   NEMO_SPEECH_BIN the nemo-speech binary (default: build/metal-server/bin/nemo-speech here)
#   SLOTS           expert cache slots per layer (default 80, about 7 GB; lower it if the Mac swaps,
#                   raise it when more memory is free)
#   MAX_CTX         longest request in tokens (default 8192, about 7 min of audio; 46000 = 1 hour)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${NEMO_SPEECH_BIN:-$HERE/../../build/metal-server/bin/nemo-speech}"
MODEL="${NEMOTRON_MODEL:-$HOME/models/nemotron-3.5-transcribe/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf}"
SLOTS="${SLOTS:-80}"
MAX_CTX="${MAX_CTX:-8192}"

[ $# -ge 1 ] || { sed -n 2,21p "$0"; exit 2; }
IN="$1"; shift

TMP=""
if [ -f "$IN" ] && ! ffprobe -v error -show_entries stream=sample_rate,channels,codec_name -of csv=p=0 "$IN" \
    | grep -qx "pcm_s16le,16000,1"; then
  TMP="$(mktemp -t nemotron).wav"
  ffmpeg -loglevel error -y -i "$IN" -ac 1 -ar 16000 -c:a pcm_s16le "$TMP"
  IN="$TMP"
fi
trap '[ -n "$TMP" ] && rm -f "$TMP"' EXIT

args=(
  transcribe "$IN"
  --model "$MODEL"
  --device metal
  --asr.backend.threads 10
  --asr.salm.moe_stream_slots "$SLOTS"
  --asr.salm.moe_stream_nocache
  --asr.salm.max_sequences 1
  --asr.salm.n_ctx_per_sequence "$MAX_CTX"
  --asr.salm.max_context "$MAX_CTX"
  --asr.salm.n_batch 2048
  --asr.salm.mtp_draft 0
  --asr.salm.encoder_max_batch 1
  --no-warmup
)
"$BIN" "${args[@]}" "$@"
