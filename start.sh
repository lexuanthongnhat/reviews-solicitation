#!/bin/bash

DATASET=$1    # accept: "edmunds", "bliu", "semeval" and "synthetic"

SCENARIO='basic'
RUN=200
POLL=300
QUESTION=3
ANSWER=''
INPUT='datasets/edmunds_reviews.csv'

if [[ $DATASET == 'edmunds' ]]; then
  COUNT_LOWBOUND=100
elif [[ $1 == 'bliu' || $1 == 'semeval' ]]; then
  COUNT_LOWBOUND=10

  ANSWER='_gen'
  if [[ ! -z "$2" && $2 == '_real' ]]; then
    ANSWER=$2
    SCENARIO='passive_vs_active'
  fi

  if [[ $1 == 'bliu' ]]; then
    INPUT='anno-datasets/bliu-datasets/'
  else
    INPUT='anno-datasets/SemEval2014-Task4/'
  fi
elif [[ $1 == 'synthetic' ]]; then
  DATASET=$1
  SCENARIO='synthetic'
  COUNT_LOWBOUND=10
  QUESTION=2
else
  echo "Unsupported dataset: $1"
  exit
fi

EXPERIMENT=${DATASET}_cl${COUNT_LOWBOUND}_poll${POLL}_q${QUESTION}_run${RUN}${ANSWER}
echo $EXPERIMENT
STEP=$(( POLL / 25 ))
LOG='INFO'
#LOG='DEBUG'

OUTPUT_DIR='output/'
python soli_start.py --input=$INPUT \
                     --dataset=$DATASET \
                     --scenario=$SCENARIO \
                     --review-count-lowbound=$COUNT_LOWBOUND \
                     --question-count=$QUESTION \
                     --poll-count=$POLL \
                     --run-count=$RUN \
                     --loglevel=$LOG \
                     --output=${OUTPUT_DIR}/${EXPERIMENT}.pickle

python visualizer.py --dataset $DATASET \
                     --experiment $EXPERIMENT \
                     --poll $(( POLL - 1)) \
                     --marker-step $STEP \
                     --scale 0.5

PLOT_DIR='plots/'
DROPBOX="${HOME}/Dropbox/testbox/review_soli/"
cp ${PLOT_DIR}/${EXPERIMENT}*.pdf $DROPBOX
