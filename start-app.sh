#!/bin/bash

# Ensure Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build and start the containers
echo "🚀 Building and starting containers..."
docker-compose up --build -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Display access information
echo ""
echo "✅ Sleep EEG Analysis App is now running!"
echo ""
echo "🌐 Frontend: http://localhost:3001"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "To stop the application, run: docker-compose down"

# Follow the logs
docker-compose logs -f
