#!/usr/bin/env bash

SCRIPT_NAME=$(basename "$0")
SHORT="e:h:v:r:pj:"
LONG="env:,hypers:,values:,exe:,progress,jobs:,help,nthreads:"

function usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTION]... -h [ARGS] -v [ARGS]

Sweep over hyperparameters in parallel.

  -e, --env FILE
    Use the specified virtual environment
  -h, --hypers ARGS
    A colon-delimited list of hyperparameters to sweep.
  -v, --values ARGS
    A comma- and colon-delimited list of hyperparameter values to sweep for
    each hyperparameter specified in the '-h' option. colons delmit values to
    sweep for each separate hyperparameter and commas delimit values to sweep
    for a specific hyperparameter. For example '1,2,3:4,5,6' will sweep values
    1, 2, and 3 for the first hyperparameter provided in the '-h' option and 4,
    5, and 6 for the second hyperparameter provided in the '-h' option.
  -r, --run FILE
    The executable to run, by default 'main.py'
  -p, --progress
    Flag which determines if the current progress is shown
  -j, --jobs num
    Set the maximum number of jobs to run in parallel to num. See the -j option
    for gnu parallel for more information:
	Run up to num jobs in parallel. Default is 100%.
	num	Run up to num jobs in parallel.
	0      	Run as many as possible (this can take a while to determine).
	num%   	Multiply the number of CPU threads by num percent. E.g. 100%
		means one job per CPU thread on each machine.
	+num	Add num to the number of CPU threads.
	-num   	Subtract num from the number of CPU threads.
    --nthreads num
	Set the value for the OMP_NUM_THREADS environment variable to 'num'. By
	default this is set to 1.
EOF
}

OPTSTRING="e:h:v:r:pj:"
HYPERS=""
VALUES=""
EXE="main.py"
PROGRESS=false
BAR=false
N_JOBS=""
OMP_NUM_THREADS=1

TEMP=$(getopt -o $SHORT --long $LONG --name "$SCRIPT_NAME" -- "$@")
eval set -- "${TEMP}"

while :; do
    case "${1}" in
        -e | --env )
	    if [[ "${2}" == *"/bin/activate" ]]; then
		source "${2}"
	    else
		source "${2}/bin/activate"
	    fi
	    shift 2
	    ;;
	-h | --hypers )
	    HYPERS="${2}"
	    shift 2
	    ;;
	-v | --values )
	    VALUES="${2}"
	    shift 2
	    ;;
	-r | --run )
	    EXE="${2}"
	    shift 2
	    ;;
	-p | --progress )
	    PROGRESS=true
	    shift
	    ;;
	-j | --jobs )
	    N_JOBS="${2}"
	    shift 2
	    ;;
	--nthreads )
	    OMP_NUM_THREADS="${2}"
	    shift 2
	    ;;
	-- )
	    shift
	    break
	    ;;
	--help )
	    usage
	    exit 0
	    break
	    ;;
        * )
	    echo "$SCRIPT_NAME: missing operand"
	    echo "Try '$SCRIPT_NAME --help' for more information"
	    exit 1
	    ;;
    esac
done

export OMP_NUM_THREADS=$OMP_NUM_THREADS

IFS=":" read -r -a hypers <<< $HYPERS
IFS=":" read -r -a values <<< $VALUES

declare -a all_hyper_sweeps

for i in ${!hypers[@]}; do
    hyper_sweeps=""
    IFS="," read -r -a individual_values <<< ${values[i]}
    for v in ${individual_values[@]}; do
	hyper_sweeps="$hyper_sweeps ${hypers[i]}=$v"
    done
    all_hyper_sweeps+=("${hyper_sweeps[@]}")
done

# Build up command
cmd="parallel"
if [[ $N_JOBS != "" ]]; then
    cmd="$cmd --jobs $N_JOBS"
fi
if $PROGRESS; then
    cmd="$cmd --eta "
fi
cmd="$cmd python3 $EXE"
for i in ${!all_hyper_sweeps[@]}; do
    cmd="$cmd :::${all_hyper_sweeps[$i]}"
done

# Print to stdout the command we are running
echo
echo -ne "\e[1;3;32m"
echo -n "Running the command: "
echo -ne "\e[0m"
echo -ne "\e[31m"
echo "$cmd"
echo -ne "\e[0m"
echo

# Run the command
eval $cmd
