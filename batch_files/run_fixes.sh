#!/bin/bash
# Run fixes and optimizations for the zone optimization pipeline

echo "Applying fixes and optimizations to Zone Optimization Pipeline..."
echo

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")

echo "Running fixes and optimizations..."
python3 "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")/utils/fixes_and_optimizations.py"

if [ $? -ne 0 ]; then
    echo
    echo "Fixes and optimizations failed with error code $?"
    exit $?
fi

echo
echo "All fixes and optimizations were applied successfully!"
echo

exit 0
