#!/bin/bash

RUN=100
STAR=6
POLL=200
ASPECT=3

PLOT_DIR='plots/'
DROPBOX="${HOME}/Dropbox/testbox/review_soli/"

#RANDOMIZE='-r'
RANDOMIZE=''
if [[ (! -z "$RANDOMIZE") && $RANDOMIZE == '-r' ]]; then
  R_SUFFIX=_random
else
  R_SUFFIX=
fi
OUTPUT=${PLOT_DIR}/doubt_star${STAR}_poll${POLL}_aspect${ASPECT}${R_SUFFIX}_run${RUN}.pdf

time python doubt_inspector.py --run-count $RUN --star-rank $STAR \
  --poll-count $POLL --aspect-count $ASPECT $RANDOMIZE --output $OUTPUT
cp $OUTPUT $DROPBOX
echo $OUTPUT
