#!/bin/bash

# enable **
shopt -s globstar

count_demos=0
count_failed=0

for p in ime_fgs/demos/**/*.py; do # loop through all demos
	DIR=$(dirname "${p}")
	FILE=$(basename "${p}")
	start_time=`date +%s`
	# run demo in corresponding folder
	OUTPUT=$(cd "$DIR"; run_headless.py "$FILE" 2>&1)
	EXIT_CODE=$?
	end_time=`date +%s`
	# print error messange if exit code is not 0
	if [ $EXIT_CODE -ne 0 ]; then
		echo -e '\033[0;31m'$p failed \($(expr $end_time - $start_time)s\)'\033[0m'
		echo $OUTPUT
		ERROR=true
		((count_failed++))
	else
		echo -e '\033[1;32m'$p successfull \($(expr $end_time - $start_time)s\)'\033[0m'
	fi
	((count_demos++))
done

count_success=$((count_demos-count_failed))
echo $count_success/$count_demos demos run successfully

# if any demo failed return 1 as exit code
if [ "$ERROR" = true ]; then
    exit 1
fi
