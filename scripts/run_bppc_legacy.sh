#!/bin/bash
### BPPC evaluation for all batter types
echo "Running BPPC for left-handed batter..."
python fine-tuning_allbaseline.py --handed left

echo "Running BPPC for right-handed batter..."
python fine-tuning_allbaseline.py --handed right

echo "All tests finished!"