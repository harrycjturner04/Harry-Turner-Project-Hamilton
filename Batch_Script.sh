#!/bin/bash
#SBATCH -p cuda
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH -t 12:00:00
#SBATCH --job-name=Harry_Turner_Project

#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=henry.c.turner@durham.ac.uk

set -euo pipefail   # exit on error, undefined var, or pipe failure

PROJECT_DIR="/home/wtsf25/Harry Turner Project"
cd "$PROJECT_DIR"

############################
# Modules
############################
module purge
module load gcc/11.2
# Do NOT `module load cuda/12.9.1` — it adds cuBLAS 12.9 to LD_LIBRARY_PATH,
# overriding torch's bundled cuBLAS 12.8 and causing CUBLAS_STATUS_INVALID_VALUE.
# Instead, set CUDA_HOME and PATH manually so FlashInfer can find nvcc for JIT
# compilation without polluting LD_LIBRARY_PATH with the module's libraries.
export CUDA_HOME=/apps/developers/compilers/cuda/12.9.1/1/default
export PATH="${CUDA_HOME}/bin:$PATH"

# FlashInfer caches JIT-compiled build.ninja files in ~/.cache/flashinfer/.
# Stale caches from previous runs may hardcode the wrong nvcc path
# (e.g. /usr/local/cuda/bin/nvcc). Clear them so FlashInfer regenerates
# build files using our CUDA_HOME.
rm -rf ~/.cache/flashinfer/ || true   # best-effort: NFS lock files may be busy

# Force linker path (fixes vLLM/Triton ld issue, but doesn't touch PATH)
export LD=/usr/bin/ld

############################
# Conda
############################
source /nobackup/wtsf25/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate /nobackup/wtsf25/anaconda3/envs/harry_turner_project

# Conda's libstdc++ must come FIRST so scipy/pyarrow get CXXABI_1.3.15.
# Because we set CUDA_HOME/PATH manually (not via module load), the cuda
# module's cuBLAS 12.9 is NOT in LD_LIBRARY_PATH, so no shadowing occurs.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"

# Workaround for cuBLAS INVALID_VALUE with BF16 GEMMs on CUDA 12.9+
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Redirect all large caches/model weights to 600GB nobackup volume
# (home quota is only 10GB — Qwen3.5-27B weights alone are ~60GB)
export HF_HOME="/nobackup/wtsf25/.cache/huggingface"
export TRANSFORMERS_CACHE="/nobackup/wtsf25/.cache/huggingface"
export UV_CACHE_DIR="/nobackup/wtsf25/.cache/uv"
export PLAYWRIGHT_BROWSERS_PATH="/nobackup/wtsf25/.cache/ms-playwright"
export PIP_CACHE_DIR="/nobackup/wtsf25/.cache/pip"

# Torch Inductor / Triton autotune caches: MUST use a persistent path.
# SLURM sets TMPDIR=/local/slurm.{JOBID}, which changes every job.
# If inductor caches reference the old TMPDIR, vLLM crashes with
# "PermissionError: Permission denied: '/local/slurm.<old_job>'".
export TORCHINDUCTOR_CACHE_DIR="/nobackup/wtsf25/.cache/torchinductor"
export TRITON_CACHE_DIR="/nobackup/wtsf25/.cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Clear stale vLLM AOT compile cache — it embeds paths from previous
# TMPDIR values. After clearing, vLLM will regenerate AOT artifacts
# using the persistent TORCHINDUCTOR_CACHE_DIR set above.
rm -rf ~/.cache/vllm/torch_compile_cache/

# vLLM must be pre-installed in the conda env via an interactive GPU session:
#   srun --pty -p cuda --gres=gpu:h200_nvl:2 -t 01:30:00 bash
#   module load gcc/11.2 cuda/12.9.1 && source /nobackup/wtsf25/anaconda3/etc/profile.d/conda.sh
#   conda activate /nobackup/wtsf25/anaconda3/envs/harry_turner_project
#   pip install 'vllm>=0.17.0'
python -c "import vllm" || { echo "ERROR: vLLM not installed. Run the interactive install first."; exit 1; }

############################
# Diagnostics
############################
echo "=== Job diagnostics ==="
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Python:     $(which python)  $(python -V 2>&1)"
echo "Conda env:  $CONDA_DEFAULT_ENV"
echo "SLURM Job:  ${SLURM_JOB_ID:-local}"
module list 2>&1
nvidia-smi || echo "nvidia-smi failed"
echo "========================"
echo ""

############################
# 1. Build Database
############################
echo "=== Step 1: Building database ==="
python Scripts/database.py
DB_STATUS=$?
if [ $DB_STATUS -ne 0 ]; then
    echo "ERROR: database.py failed with exit code $DB_STATUS"
    exit $DB_STATUS
fi
echo "✓ Database built successfully"
echo ""

############################
# 2. Start vLLM Server
############################
# Derive a unique port from SLURM_JOB_ID to avoid collisions
VLLM_PORT=$(( 8000 + ${SLURM_JOB_ID:-0} % 1000 ))

# Load Hugging Face API key from .env (if present). Avoid echoing the token.
if [ -f .env ]; then
    HUGGINGFACE_API_KEY=$(grep -E '^HUGGINGFACE_API_KEY=' .env | tail -n 1 | cut -d '=' -f2- || true)
    export HUGGINGFACE_API_KEY="${HUGGINGFACE_API_KEY:-}"
else
    export HUGGINGFACE_API_KEY="${HUGGINGFACE_API_KEY:-}"
fi

# vLLM readiness timeout (used by ServerManager inside pipeline)
export VLLM_READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-600}"

# Enforce deeper analytics: minimum plots required for analysis stage
export PIPELINE_MIN_PLOTS_ANALYSIS="${PIPELINE_MIN_PLOTS_ANALYSIS:-3}"

# Per-GroupChat wall-clock timeout (seconds).  Matches context.md recommendation.
export PIPELINE_CHAT_TIMEOUT_S="${PIPELINE_CHAT_TIMEOUT_S:-900}"

echo "=== Step 2: Starting vLLM server on port $VLLM_PORT ==="

# Prefer passing tokens via env so they don't appear in logs/process lists.
export HF_TOKEN="$HUGGINGFACE_API_KEY"
export HUGGING_FACE_HUB_TOKEN="$HUGGINGFACE_API_KEY"

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-27B \
  --port "$VLLM_PORT" \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 4}' \
  --override-generation-config '{"chat_template_kwargs": {"enable_thinking": false}}' &

VLLM_PID=$!
echo "Started vLLM server with PID $VLLM_PID on port $VLLM_PORT"

# --- helper: kill vLLM on exit regardless of success/failure ---
cleanup() {
    # Kill any vLLM servers that might be running (text or VLM)
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        echo "Killed vLLM server (PID $VLLM_PID)"
    fi
}
trap cleanup EXIT

# Wait for the server to become ready.
# NOTE: We use Python's urllib instead of curl because LD_LIBRARY_PATH
# (conda's libcurl.so) poisons the system curl binary, causing silent failures.
echo "Waiting for vLLM server to be ready..."
echo "  (First run with a new model may take several minutes to download weights)"
sleep 120

SERVER_READY=false
for i in {1..20}; do
    # Check if vLLM process is still alive
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server process (PID $VLLM_PID) died unexpectedly."
        echo "Check the output above for vLLM error messages."
        wait "$VLLM_PID" 2>/dev/null
        VLLM_EXIT=$?
        echo "vLLM exit code: $VLLM_EXIT"
        exit 1
    fi
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:${VLLM_PORT}/v1/models')" 2>/dev/null; then
        echo "vLLM server is ready!"
        SERVER_READY=true
        break
    fi
    echo "  Waiting for server... (attempt $i/20, vLLM PID $VLLM_PID still alive)"
    sleep 60
done

if [ "$SERVER_READY" = false ]; then
    echo "ERROR: vLLM server did not become ready in time."
    exit 1
fi

python -c "import urllib.request, sys; sys.stdout.buffer.write(urllib.request.urlopen('http://localhost:${VLLM_PORT}/v1/models').read()); print()"
echo ""

############################
# 3. Run Captain Pipeline
############################
# Point OpenAI-compatible clients to local vLLM
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
# Export server info so Python ServerManager can adopt & swap models
export VLLM_PID="$VLLM_PID"
export VLLM_PORT="$VLLM_PORT"

# Pipeline execution mode for ablation study.
# Override at submission time: PIPELINE_MODE=single_pass sbatch Batch_Script.sh
PIPELINE_MODE="${PIPELINE_MODE:-full_closed_loop}"
export PIPELINE_MODE
echo "Pipeline mode: $PIPELINE_MODE"

# Install reportlab for PDF report generation (non-fatal if unavailable)
pip install --quiet reportlab || echo "WARN: reportlab not installed — PDF output will be skipped"

# Generate unique run ID
RUN_ID="slurm_${SLURM_JOB_ID:-0}_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=== Step 3: Running CaptainAgent pipeline ==="
echo "Run ID:        $RUN_ID"
echo "Model:         Qwen/Qwen3.5-27B"
echo "Base URL:      $OPENAI_BASE_URL"
echo "Pipeline mode: $PIPELINE_MODE"
echo ""

python Scripts/main.py \
    "Database/raw" \
    "Outputs/$RUN_ID" \
    --model          "Qwen/Qwen3.5-27B" \
    --temperature    0.0 \
    --api-key        "$OPENAI_API_KEY" \
    --base-url       "$OPENAI_BASE_URL" \
    --metadata-db    "Database/metadata" \
    --context-path   "context.md" \
    --log-level      INFO \
    --pipeline-mode  "$PIPELINE_MODE"

PIPELINE_STATUS=$?

if [ $PIPELINE_STATUS -ne 0 ]; then
    echo "ERROR: Multi-agent pipeline failed with exit code $PIPELINE_STATUS"
    exit $PIPELINE_STATUS
fi

echo ""
echo "✓ Pipeline completed successfully"
echo "  Output:  Outputs/$RUN_ID"
echo "  Job finished on $(date)"
