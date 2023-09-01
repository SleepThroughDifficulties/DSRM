TIME="`date +%m%d%H%M`"

# HYPER PARA
DATASET_NAME="imdb"
TASK_NAME="None"
EPOCHS=10
MAX_SEQ_LENGTH=256
BSZ=8
META=1
SEED=46
CLAMP="0.1"
CUDA=0

for each_clamp in $CLAMP
do
      echo "COMMAND: Date${TIME}_ds${DATASET_NAME}_epochs${EPOCHS}_seed${SEED}_meta${META}_clamp${each_clamp}"
      
      python -u upper_train.py \
            --date $TIME \
            --shifting $META \
            --dataset_name $DATASET_NAME \
            --task_name $TASK_NAME \
            --epochs $EPOCHS \
            --max_seq_length $MAX_SEQ_LENGTH \
            --bsz $BSZ \
            --cuda $CUDA \
            --loss_clamp $each_clamp \
            --seed $SEED
done