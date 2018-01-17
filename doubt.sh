#!/bin/bash

RUN=1000
STAR=6
USER=200
ASPECT=6
RANDOMIZE='-r'
#RANDOMIZE=

if [ $RANDOMIZE == '-r' ]
then
  R_SUFFIX=_random
else
  R_SUFFIX=
fi
OUTPUT=plots/plot_run${RUN}_star${STAR}_user${USER}_aspect${ASPECT}${R_SUFFIX}.pdf

time python doubt_inspector.py --run-count $RUN --star-rank $STAR \
  --user-count $USER --aspect-count $ASPECT $RANDOMIZE --output $OUTPUT
echo $OUTPUT
