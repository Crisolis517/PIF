#!/usr/bin/env bash
# Servicios locales SIN Docker (entorno usado para generar el entregable).
# Uso: scripts/servicios_locales.sh start|stop|status
# Requiere binarios de PostgreSQL (PG_BIN) y un mongod portable (MONGOD).
set -euo pipefail
cd "$(dirname "$0")/.."
PG_BIN="${PG_BIN:-/Library/PostgreSQL/18/bin}"
MONGOD="${MONGOD:-tools/mongodb-macos-aarch64-8.0.4/bin/mongod}"
case "${1:-status}" in
  start)
    [ -d .pgdata ] || "$PG_BIN/initdb" -D .pgdata -U pi_user --auth=trust -E UTF8 --locale=C >/dev/null
    "$PG_BIN/pg_ctl" -D .pgdata -o "-p 5433 -k /tmp" -l logs/postgres_server.log start
    sleep 2
    "$PG_BIN/createdb" -h localhost -p 5433 -U pi_user proyecto_integrador 2>/dev/null || true
    mkdir -p .mongodata
    "$MONGOD" --dbpath .mongodata --port 27018 --bind_ip 127.0.0.1 --fork --logpath logs/mongod_server.log
    echo "Listo: PG_PORT=5433, MONGO_URI=mongodb://127.0.0.1:27018 (ajustar .env)";;
  stop)
    "$PG_BIN/pg_ctl" -D .pgdata stop || true
    pkill -TERM -f "mongod --dbpath .mongodata" || true;;
  status)
    "$PG_BIN/pg_isready" -h localhost -p 5433 || true
    pgrep -fl "mongod --dbpath .mongodata" || echo "mongod detenido";;
esac
