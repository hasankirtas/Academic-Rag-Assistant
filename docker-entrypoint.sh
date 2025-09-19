#!/bin/bash
set -e

# Academic RAG Assistant Docker Entrypoint Script

echo "🚀 Starting Academic RAG Assistant..."

# Function to handle shutdown gracefully
cleanup() {
    echo "🛑 Shutting down gracefully..."
    # Kill any background processes
    jobs -p | xargs -r kill
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Create necessary directories with proper permissions
echo "📁 Creating directories..."
mkdir -p /app/data/raw /app/data/processed /app/data/output /app/logs /app/cache
chmod 755 /app/data /app/logs /app/cache
chmod 755 /app/data/raw /app/data/processed /app/data/output

# Set proper ownership (in case of volume mounts)
if [ "$(id -u)" = "0" ]; then
    echo "🔧 Setting ownership for mounted volumes..."
    chown -R appuser:appuser /app/data /app/logs /app/cache 2>/dev/null || true
fi

# Note: Hugging Face token can be entered directly in the UI
echo "ℹ️  Hugging Face API token can be entered in the Streamlit UI"

# Display configuration
echo "📋 Configuration:"
echo "   - Streamlit Port: ${STREAMLIT_SERVER_PORT:-8501}"
echo "   - Server Address: ${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
echo "   - Headless Mode: ${STREAMLIT_SERVER_HEADLESS:-true}"
echo "   - Log Level: ${LOG_LEVEL:-INFO}"

# Wait for any initialization if needed
if [ -n "$WAIT_FOR_SERVICES" ]; then
    echo "⏳ Waiting for external services..."
    # Add service health checks here if needed
fi

# Start the application
echo "🎯 Starting Streamlit application..."
exec streamlit run src/inference/streamlit_ui.py \
    --server.port=${STREAMLIT_SERVER_PORT:-8501} \
    --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    --server.headless=${STREAMLIT_SERVER_HEADLESS:-true} \
    --browser.gatherUsageStats=${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false} \
    --server.enableCORS=${STREAMLIT_SERVER_ENABLE_CORS:-false} \
    --server.enableXsrfProtection=${STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION:-true} \
    --logger.level=${LOG_LEVEL:-info}
