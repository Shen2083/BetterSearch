#!/usr/bin/env bash
# Stop and remove the node. The daemon is left running.
docker rm -f vespa 2>/dev/null || true
echo "vespa removed"
