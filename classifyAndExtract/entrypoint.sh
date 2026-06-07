#!/usr/bin/env bash
# Start SSH daemon in the background, then run the container's main command.
set -e

mkdir -p /var/run/sshd
/usr/sbin/sshd

exec "$@"
