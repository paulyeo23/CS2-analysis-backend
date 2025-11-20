param(
    [string]$SourceDir
)

# Build the Docker image
docker build -t simple-uploader -f docker/Dockerfile .

# Run it, mounting credentials + source code
docker run `
    -e GOOGLE_APPLICATION_CREDENTIALS="/secrets/key.json" `
    -v "C:/users/njadeed/project/secrets/key.json:/secrets/key.json" `
    -v "C:/users/njadeed/project/replays_upload/app:/app" `
    -v "${SourceDir}:/data" `
    simple-uploader "/data"