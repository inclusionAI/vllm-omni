#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../../../.." && pwd)"
: "${MODEL:?Set MODEL to the LingV3 checkpoint directory}"

exec vllm-omni serve "$MODEL" \
    --omni \
    --deploy-config "${DEPLOY_CONFIG:-$ROOT/vllm_omni/deploy/lingv3_tts.yaml}" \
    --served-model-name "${SERVED_MODEL_NAME:-lingv3-tts}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8091}" \
    "$@"
