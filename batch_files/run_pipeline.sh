#!/bin/bash
# Main pipeline execution script for Linux
# This file runs the complete zone optimization pipeline

echo "Starting Zone Optimization Pipeline..."
echo

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")

# Create timestamp for output directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output directories
OUTPUT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")/output/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/visualizations"
mkdir -p "${OUTPUT_DIR}/results"

# Set config file path
CONFIG_FILE="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")/config/pipeline_config.ini"

echo "Using configuration file: ${CONFIG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo

echo "Running pipeline..."
python3 "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")/pipeline.py" --config "${CONFIG_FILE}"

if [ $? -ne 0 ]; then
    echo
    echo "Pipeline execution failed with error code $?"
    exit $?
fi

echo
echo "Pipeline execution completed successfully!"
echo "Results are available in: ${OUTPUT_DIR}"
echo

exit 0
