#!/usr/bin/env bash
# Deploy gateway + executor + redis to a local kind cluster (model server = compose Ollama on the host GPU).
set -euo pipefail
KIND=${KIND:-$(command -v kind || echo ~/go/bin/kind)}
cd "$(dirname "$0")/.."
docker compose build gateway executor
docker pull -q redis:7.4-alpine >/dev/null
$KIND get clusters | grep -qx sqlforge || $KIND create cluster --config k8s/kind-config.yaml
# `kind load` imports with --all-platforms, which fails for multi-arch images whose other platforms are not
# stored locally; import the current platform only.
for img in sqlforge-gateway:latest sqlforge-py:latest redis:7.4-alpine; do
  docker save "$img" | docker exec -i sqlforge-control-plane ctr --namespace=k8s.io images import --snapshotter=overlayfs -
done
GW=$(docker network inspect kind -f '{{range .IPAM.Config}}{{.Gateway}} {{end}}' | tr ' ' '\n' | grep -v ':' | grep -m1 .)
kubectl kustomize k8s/overlays/kind | sed "s/HOST_GATEWAY_IP/${GW}/" | kubectl --context kind-sqlforge apply -f -
kubectl --context kind-sqlforge -n sqlforge rollout status deploy/redis deploy/executor deploy/gateway --timeout=180s
echo "gateway: http://localhost:18080/ask"
