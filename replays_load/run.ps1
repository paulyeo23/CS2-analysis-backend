param(
    [string]$SourceDir
)

$SECRET_FILE = Join-Path $SourceDir "key.json"
echo "mount at $SECRET_FILE"
docker build -t sql-loader -f docker/Dockerfile .

docker run `
    -e GOOGLE_APPLICATION_CREDENTIALS="/secrets/key.json" `
    -e DB_HOST="34.150.184.220" `
    -e DB_NAME="replays-data" `
    -e DB_USER="postgres" `
    -e DB_PASSWORD="cse6242-dbase" `
    -v "C:/project/secrets/key.json:/secrets/key.json" `
    sql-loader "/data"