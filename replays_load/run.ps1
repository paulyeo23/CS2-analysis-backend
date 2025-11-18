param(
    [string]$SourceDir
)

docker build -t sql-loader -f docker/Dockerfile .

docker run `
    -e GOOGLE_APPLICATION_CREDENTIALS="/secrets/key.json" `
    -e DB_HOST="34.150.184.220" `
    -e DB_NAME="replays-dbe" `
    -e DB_USER="postgres" `
    -e DB_PASSWORD="+$3s6L)EFqUf`zKX" `
    -v "C:\project\secrets\key.json:/secrets/key.json" `
    sql-loader "/data"