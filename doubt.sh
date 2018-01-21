#!/bin/bash

RUN=1000
STAR=6
USER=200
ASPECT=3

PLOT_DIR='plots/'
DROPBOX="${HOME}/Dropbox/testbox/review_soli/"

#RANDOMIZE='-r'
RANDOMIZE=' '
if [ $RANDOMIZE == '-r' ]
then
  R_SUFFIX=_random
else
  R_SUFFIX=
fi
OUTPUT=${PLOT_DIR}/doubt_star${STAR}_user${USER}_aspect${ASPECT}${R_SUFFIX}_run${RUN}.pdf

time python doubt_inspector.py --run-count $RUN --star-rank $STAR \
  --user-count $USER --aspect-count $ASPECT $RANDOMIZE --output $OUTPUT
cp $OUTPUT $DROPBOX
echo $OUTPUT
