#!/bin/bash
# Run tests for the zone optimization pipeline

echo "Running tests for Zone Optimization Pipeline..."
echo

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")

# Create test output directory
TEST_OUTPUT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")/test_output"
mkdir -p "${TEST_OUTPUT_DIR}"

echo "Running tests..."
python3 "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")/tests/test_pipeline.py"

if [ $? -ne 0 ]; then
    echo
    echo "Tests failed with error code $?"
    exit $?
fi

echo
echo "All tests passed successfully!"
echo "Test results are available in: ${TEST_OUTPUT_DIR}"
echo

exit 0
