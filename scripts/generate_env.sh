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

# Find the file containing the MMIRAGE repository, to do so
# (1) Check if this folder is a git folder containing remote with string *EPFLiGHT/MMIRAGE*
# (2) Check if subdirectory (depth = 1) does in case of failure
# If both fail, fail the script
MMIRAGE_PATH=""

check_mmirage_folder() {
    if [ -d "$1" ] && [ -d "$1/.git" ]; then
        REMOTE_NAME=$($GIT_EXEC -C $1 remote get-url origin)
        if [[ "$REMOTE_NAME" =~ "EPFLiGHT/MMIRAGE" ]] || [[ "$REMOTE_NAME" =~ "mmirage" ]]; then
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}

# Check if current folder is MMIRAGE root folder
if $GIT_EXEC rev-parse --is-inside-work-tree 1>/dev/null 2>/dev/null; then
    ROOT_DIR=$($GIT_EXEC rev-parse --show-toplevel)
    check_mmirage_folder "$ROOT_DIR"
    if [ $? -eq 0 ]; then
        MMIRAGE_PATH="$ROOT_DIR"
    fi
fi

# If we failed (aka. MMIRAGE_PATH is still "") then we try to do the same for each subfolder of
# the current directory
if [ -z "$MMIRAGE_PATH" ]; then
    for subfolder in $(find . -maxdepth 1 -type d); do
        check_mmirage_folder "$subfolder"
        if [ $? -eq 0 ]; then
            MMIRAGE_PATH="$subfolder"
            break
        fi
    done
fi

# If we still did not found, then cancel
if [ -z "$MMIRAGE_PATH" ]; then
    printf "${COLOR_RED}Failed to find a valid path to the MMIRAGE root folder$COLOR_RESET\n" 1>&2
    printf "${COLOR_RED}You should consider cloning the repository first and running this script within the repository$COLOR_RESET\n" 1>&2
    exit 1
fi
MMIRAGE_PATH=$(realpath $MMIRAGE_PATH)
printf "${COLOR_GREEN}Found the path to the MMIRAGE repository at $MMIRAGE_PATH\n${COLOR_RESET}"

# Generate the .edf toml file if detected on the CSCS
if [[ $(hostname) =~ "clariden" ]] || [[ $(hostname) =~ "nid" ]]; then
    # Check if there is a file at $ENV_EDF_PATH
    ENV_EDF_PATH="/users/$USER/.edf/mmirage.toml"
    ENV_EDF_CONTENT=$(cat <<EOF
image = "docker.io/lmsysorg/sglang:latest"
mounts = ["/capstor", "/iopsstor", "/users"]

writable = true
workdir = "/users/$USER"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"

[env]
NCCL_NET = "AWS Libfabric"
NCCL_CROSS_NIC = "1"
NCCL_NET_GDR_LEVEL = "PHB"
FI_CXI_DISABLE_HOST_REGISTER = "1"
FI_MR_CACHE_MONITOR = "userfaultfd"
FI_CXI_DEFAULT_CQ_SIZE = "131072"
FI_CXI_DEFAULT_TX_SIZE = "32768"
FI_CXI_RX_MATCH_MODE = "software"
FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD = "16777216"
FI_CXI_COMPAT = "0"

EOF
)
    
    # If the file already exists, git diff against the expected value
    should_generate=1
    if test -f $ENV_EDF_PATH; then
        should_generate=0

        # Create a temporary
        if ! diff $ENV_EDF_PATH - <<< "$ENV_EDF_CONTENT" > /dev/null; then
            printf "${COLOR_YELLOW}It appears that the $ENV_EDF_PATH already exists.${COLOR_RESET}\n"
            printf "${COLOR_YELLOW}Do you want to overwrite it? (y/n)${COLOR_RESET} "
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                should_generate=1
            fi
        fi
    fi

    if [ $should_generate -eq 1 ]; then
        mkdir -p $(dirname $ENV_EDF_PATH)
        echo "$ENV_EDF_CONTENT" > $ENV_EDF_PATH
        printf "${COLOR_GREEN}Successfully generated the .edf file at $ENV_EDF_PATH${COLOR_RESET}\n"
    fi
fi

# Generate the .env file
ENV_FILE_PATH="$MMIRAGE_PATH/.env"
ENV_FILE_CONTENT=$(cat <<EOF
# Auto-generated environment file for MMIRAGE
MMIRAGE_PATH=$MMIRAGE_PATH
EOF
)

should_generate=1
if test -f $ENV_FILE_PATH; then
    should_generate=0
    if ! diff $ENV_FILE_PATH - <<< "$ENV_FILE_CONTENT" > /dev/null; then
        printf "${COLOR_YELLOW}It appears that the $ENV_FILE_PATH already exists.${COLOR_RESET}\n"
        printf "${COLOR_YELLOW}Do you want to overwrite it? (y/n)${COLOR_RESET} "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            should_generate=1
        fi
    fi
fi

if [ $should_generate -eq 1 ]; then
    echo "$ENV_FILE_CONTENT" > $ENV_FILE_PATH
    printf "${COLOR_GREEN}Successfully generated the .env file at $ENV_FILE_PATH${COLOR_RESET}\n"
fi

printf "${COLOR_GREEN}Setup complete!${COLOR_RESET}\n"
