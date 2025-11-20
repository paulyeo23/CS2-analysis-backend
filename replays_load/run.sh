#!/bin/bash

# create the docker image:
docker build -t sql-loader -f docker/Dockerfile .

# path to secrets 
SECRETS_PATH="$(dirname "$(pwd)")/secrets"
KEY_FILE="$SECRETS_PATH/key.json"

SECRETS_PATH_WINDOWS=$(wslpath -w "$KEY_FILE")


echo "Mounting Secrets from:"
echo "Linux path: $SECRETS_PATH_WINDOWS"

docker run \
    -e GOOGLE_APPLICATION_CREDENTIALS="/secrets/key.json" \
    -e DB_HOST="34.150.184.220" \
    -e DB_NAME="replays-dbe" \
    -e DB_USER="postgres" \
    -e DB_PASSWORD="cse6242-dbase" \
    -v "$KEY_FILE:/secrets/key.json" \
    sql-loader "/data"