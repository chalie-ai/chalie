#!/bin/bash
# Helper script to run memory pipeline tests from remote server
# Usage: ./run_memory_test.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Memory Pipeline Test Runner${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if running inside docker container
if [ ! -f /.dockerenv ]; then
    echo -e "${YELLOW}Not running inside Docker container${NC}"
    echo "Attempting to execute inside python-agent container..."
    docker exec -it python-agent bash -c "cd /app && ./run_memory_test.sh"
    exit 0
fi

# We're inside the container
echo -e "${GREEN}Running inside Docker container${NC}"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if test file exists
if [ ! -f "test_memory_pipeline.py" ]; then
    echo -e "${RED}Error: test_memory_pipeline.py not found${NC}"
    exit 1
fi

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="/tmp/memory-test-${TIMESTAMP}.log"

echo -e "Test script: ${GREEN}test_memory_pipeline.py${NC}"
echo -e "Log file: ${GREEN}${LOG_FILE}${NC}"
echo ""

# Check if consumer is running
echo -e "${YELLOW}Checking if consumer is running...${NC}"
if ps aux | grep -v grep | grep "consumer.py" > /dev/null; then
    echo -e "${GREEN}✓ Consumer is running${NC}"
else
    echo -e "${RED}✗ Consumer is NOT running${NC}"
    echo -e "${YELLOW}Please start consumer.py before running tests${NC}"
    echo "Start with: python3 consumer.py &"
    exit 1
fi
echo ""

# Show available test messages
echo -e "${YELLOW}Test messages:${NC}"
echo "  1. Hello! How are you today?"
echo "  2. Can you remember my favorite color is blue?"
echo "  3. What's the weather like?"
echo "  4. I'm feeling happy today because I got a promotion!"
echo "  5. Could you help me understand how episodic memory works?"
echo ""

# Confirm execution
echo -e "${YELLOW}WARNING: Each test takes 5-15 minutes (episodic consolidation delay)${NC}"
echo -e "${YELLOW}Total estimated time: ~45-75 minutes for all 5 tests${NC}"
echo ""
read -p "Continue with test execution? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Test execution cancelled"
    exit 0
fi

echo ""
echo -e "${GREEN}Starting test suite...${NC}"
echo ""

# Run the test
python3 test_memory_pipeline.py 2>&1 | tee "${LOG_FILE}"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Test suite completed successfully${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "Full log saved to: ${GREEN}${LOG_FILE}${NC}"
    echo ""
    echo "To view the log:"
    echo "  cat ${LOG_FILE}"
    echo ""
    echo "To view only errors:"
    echo "  grep ERROR ${LOG_FILE}"
    echo ""
    echo "To view summary:"
    echo "  grep -A 20 'TEST SUITE SUMMARY' ${LOG_FILE}"
else
    echo ""
    echo -e "${RED}================================${NC}"
    echo -e "${RED}Test suite failed${NC}"
    echo -e "${RED}================================${NC}"
    echo ""
    echo -e "Check log for details: ${RED}${LOG_FILE}${NC}"
    echo ""
    exit 1
fi
