#!/bin/bash

# Constant throughout the file
GIT_EXEC="git"

COLOR_RESET='\033[0m'
COLOR_RED='\033[31m'
COLOR_GREEN='\033[32m'
COLOR_YELLOW='\033[33m'
COLOR_BLUE='\033[34m'
COLOR_MAGENTA='\033[35m'
COLOR_CYAN='\033[36m'
COLOR_WHITE='\033[37m'

# Find the file containing the MIRAGE repository, to do so
# (1) Check if this folder is a git folder containing remote with string *EPFLiGHT/MIRAGE*
# (2) Check if subdirectory (depth = 1) does in case of failure
# If both fail, fail the script
MIRAGE_PATH=""

check_mirage_folder() {
    if [ -d "$1" ] && [ -d "$1/.git" ]; then
        REMOTE_NAME=$($GIT_EXEC -C $1 remote get-url origin)
        if [[ "$REMOTE_NAME" =~ "EPFLiGHT/MIRAGE" ]]; then
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}

# Check if current folder is MIRAGE root folder
if $GIT_EXEC rev-parse --is-inside-work-tree 1>/dev/null 2>/dev/null; then
    ROOT_DIR=$($GIT_EXEC rev-parse --show-toplevel)
    check_mirage_folder "$ROOT_DIR"
    if [ $? -eq 0 ]; then
        MIRAGE_PATH="$ROOT_DIR"
    fi
fi

# If we failed (aka. MIRAGE_PATH is still "") then we try to do the same for each subfolder of
# the current directory
if [ -z "$MIRAGE_PATH" ]; then
    for subfolder in $(find . -maxdepth 1 -type d); do
        check_mirage_folder "$subfolder"
        if [ $? -eq 0 ]; then
            MIRAGE_PATH="$subfolder"
            break
        fi
    done
fi

# If we still did not found, then cancel
if [ -z "$MIRAGE_PATH" ]; then
    printf "${COLOR_RED}Failed to find a valid path to the MIRAGE root folder$COLOR_RESET\n" 1>&2
    printf "${COLOR_RED}You should consider cloning the repository first and running this script within the repository$COLOR_RESET\n" 1>&2
    exit 1
fi
MIRAGE_PATH=$(realpath $MIRAGE_PATH)
printf "${COLOR_GREEN}Found the path to the MIRAGE repository at $MIRAGE_PATH\n${COLOR_RESET}"

# Generate the .env based on the retrieved configuration
OUTPUT_PATH="$MIRAGE_PATH/.env"
OUTPUT_TEXT=$(cat <<EOF
# This .env file has been generated programmatically using the 
# script generate_env.sh

MIRAGE_PATH="$MIRAGE_PATH"
\n\n
EOF
)

printf "${COLOR_GREEN}The .env file has been generated successfully at $OUTPUT_PATH\n${COLOR_RESET}"
printf "$OUTPUT_TEXT" > $OUTPUT_PATH


