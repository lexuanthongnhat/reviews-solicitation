#!/bin/bash

debug=""

if [ $2 == "-debug" ]; then
  debug="-m pdb"
fi

if [ $1 == "edmunds" ]; then
  python $debug soli_start.py --input=datasets/edmunds_reviews.csv \
                              --question-count=2 \
                              --review-count-lowbound=100 \
                              --poll-count=300 \
                              --run-count=200 \
                              --output=output/edmunds_l100_p300_q2_r200.pickle \
                              --loglevel=DEBUG
elif [ $1 == "bliu" ]; then
  python $debug soli_start.py --input=anno-datasets/bliu-datasets/ \
                              --dataset=bliu \
                              --question-count=3 \
                              --scenario=basic \
                              --review-count-lowbound=10 \
                              --poll-count=300 \
                              --run-count=200 \
                              --output=output/bliu_p300_q3_r200_gen.pickle \
                              --loglevel=DEBUG
else
  python $debug soli_start.py --input=anno-datasets/SemEval2014-Task4/ \
                              --dataset=semeval \
                              --question-count=2 \
                              --review-count-lowbound=10 \
                              --poll-count=300 \
                              --run-count=2 \
                              --output=output/semeval_l10_p300_q2_2.pickle \
                              --loglevel=DEBUG

fi
