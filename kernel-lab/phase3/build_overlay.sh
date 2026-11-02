#!/usr/bin/env bash
# Build the Phase 3 experimental image: the serving image plus patches/0020 applied to the
# installed vllm_exl3 python. Nothing is recompiled -- the compiled extensions, the base
# image, the pins and the image LABELS are inherited, so `./m31.sh start`'s label check
# still passes and the only delta is the planner module plus one call site.
#
#   kernel-lab/phase3/build_overlay.sh [tag]
#
# The functional image is never overwritten: the tag defaults to <base>-a3. Built on both
# ranks from the same two files (the base image already exists on both), which avoids a
# 29 GiB docker save across the cable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${LAB_BASE_IMAGE:-m31-flash-exl3-tp2:053c634d-8d9c23f6}"
TAG="${1:-${BASE}-a3}"
SRC=/home/rodman/build/m31/src/vllm-exl3
CTX="$(mktemp -d /tmp/m31-a3-ctx.XXXXXX)"
trap 'rm -rf "$CTX"' EXIT

# Apply the patch to a pristine copy of the pinned, already-patched plugin tree, so the
# overlay is exactly what patches/0020 produces -- not a hand-edited file. The patch is
# rooted at the plugin root (paths are src/vllm_exl3/...), so copy the whole tree.
WORK="$(mktemp -d /tmp/m31-a3-work.XXXXXX)"
trap 'rm -rf "$CTX" "$WORK"' EXIT
tar -C "$SRC" --exclude=.git --exclude=__pycache__ -cf - . | tar -C "$WORK" -xf -
git -C "$WORK" init -q
git -C "$WORK" add -A >/dev/null
git -C "$WORK" -c user.name=kernel-lab -c user.email=kernel-lab@localhost commit -qm base
git -C "$WORK" apply "$ROOT/patches/0020-minimax-exl3-q4-slice-planner-v2.patch"
echo "patch 0020 applied cleanly"

cp "$WORK/src/vllm_exl3/slice_planner.py" "$CTX/"
cp "$WORK/src/vllm_exl3/minimax_q4kv.py" "$CTX/"

cat > "$CTX/Dockerfile" <<EOF
ARG BASE=$BASE
FROM \$BASE
# patches/0020: the O(queries x routes) q4 prefill slice planner, off by default
# (VLLM_EXL3_Q4KV_PLANNER_V2). Pure python; the compiled extensions are inherited.
COPY slice_planner.py /usr/local/lib/python3.12/dist-packages/vllm_exl3/slice_planner.py
COPY minimax_q4kv.py  /usr/local/lib/python3.12/dist-packages/vllm_exl3/minimax_q4kv.py
RUN python3 -c "import hashlib,sys; \\
    import importlib.util as u; \\
    spec=u.spec_from_file_location('sp','/usr/local/lib/python3.12/dist-packages/vllm_exl3/slice_planner.py'); \\
    m=u.module_from_spec(spec); sys.modules['vllm_exl3.slice_planner']=m; \\
    src=open('/usr/local/lib/python3.12/dist-packages/vllm_exl3/minimax_q4kv.py').read(); \\
    assert 'plan_prefill_slices_v2' in src and '_planner_v2_enabled' in src, 'patch 0020 missing'; \\
    print('overlay verified', hashlib.sha256(open('/usr/local/lib/python3.12/dist-packages/vllm_exl3/slice_planner.py','rb').read()).hexdigest()[:16])"
EOF

echo "building $TAG on rank 0"
docker build -t "$TAG" "$CTX"

NODES_R1="${LAB_RANK1:-10.100.24.1}"
SSH_USER="${SSH_USER:-rodman}"
echo "building $TAG on rank 1 ($NODES_R1)"
ssh -o BatchMode=yes "$SSH_USER@$NODES_R1" "mkdir -p /tmp/m31-a3-ctx"
scp -q -o BatchMode=yes "$CTX"/* "$SSH_USER@$NODES_R1:/tmp/m31-a3-ctx/"
ssh -o BatchMode=yes "$SSH_USER@$NODES_R1" "docker build -t '$TAG' /tmp/m31-a3-ctx && rm -rf /tmp/m31-a3-ctx"

for r in 0 1; do
  node="${NODES_R1}"; [[ $r == 0 ]] && node=localhost
  if [[ $r == 0 ]]; then
    docker image inspect "$TAG" --format "rank0 {{.Id}} labels={{index .Config.Labels \"local.m31.patch-sha256\"}}"
  else
    ssh -o BatchMode=yes "$SSH_USER@$NODES_R1" "docker image inspect '$TAG' --format 'rank1 {{.Id}} labels={{index .Config.Labels \"local.m31.patch-sha256\"}}'"
  fi
done
echo "built $TAG on both ranks; the base image $BASE is untouched"
