#!/usr/bin/env bash
# Bring up a Vespa node from nothing. Idempotent.
#
# There is no systemd here and no running Docker daemon at session start, so
# both have to be started by hand. Bridge networking with published ports is
# deliberate: with --network host the container inherits the host's hostname,
# which does not match VESPA_CONFIGSERVERS, and ZooKeeper then cannot work out
# its own id (the config server restart-loops on "zookeeper-server must be
# initialized: [myid]"). The container needs no outbound network of its own:
# models are bundled into the application package and the feed comes from
# localhost, which matters because processes inside containers cannot reach
# this environment's outbound proxy.
set -euo pipefail

IMAGE="ghcr.io/vespa-engine/vespa@sha256:f9cef4b0eb8de08a2d3d56a01377dfdb7f71a9747dfb0f610915bc608e82dd97"
APP_DIR="$(cd "$(dirname "$0")/../app" && pwd)"

if [ ! -S /var/run/docker.sock ]; then
  echo "starting dockerd"
  mkdir -p /home/user/.docker-data
  nohup dockerd --data-root=/home/user/.docker-data --storage-driver=overlay2 \
    > /tmp/dockerd.log 2>&1 &
  for _ in $(seq 1 30); do [ -S /var/run/docker.sock ] && break; sleep 1; done
fi

# Docker Hub rate-limits anonymous pulls from this egress IP; ghcr does not.
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

if ! docker ps --format '{{.Names}}' | grep -qx vespa; then
  docker rm -f vespa >/dev/null 2>&1 || true
  docker run --detach --name vespa --hostname vespa-container --memory 10g \
    --publish 8080:8080 --publish 19071:19071 --publish 19050:19050 "$IMAGE" >/dev/null
fi

wait_for() {
  local port=$1 name=$2
  for _ in $(seq 1 80); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' \
            "http://localhost:$port/state/v1/health")" = "200" ]; then
      echo "$name ready"; return 0
    fi
    sleep 3
  done
  echo "$name did not come up; docker logs vespa" >&2; return 1
}

wait_for 19071 "config server"

rm -f /tmp/app.zip
(cd "$APP_DIR" && zip -q -r /tmp/app.zip .)
curl --noproxy '*' -sf --header Content-Type:application/zip --data-binary @/tmp/app.zip \
  http://localhost:19071/application/v2/tenant/default/prepareandactivate \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("message", d))'

wait_for 8080 "container"
