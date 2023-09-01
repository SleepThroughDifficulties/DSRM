
#MODEL_DIR="Date09061147_dsimdb_epochs10_seed46_meta1_clamp0.12"
MODEL_DIR=$1
CUDA=1
ATTACK_METHOD="textbugger bertattack textfooler"
#echo"test"
regex() {
    [[ $1 =~ $2 ]] && printf '%s\n' "${BASH_REMATCH}"
}

get_task_name() {
if [ "$1" == "glue" ]; then
    TASK_NAME="sst2"
else
   TASK_NAME="None"
fi
}

#liuyan 0910 20：20
get_num_examples() {
if [ "$1" == "glue" ]; then
    NUM_EXAMPLES=872
else
   NUM_EXAMPLES=1000
fi
}


regex $MODEL_DIR 'Date([0-9]+)_ds([a-z｜_]+)_epochs([0-9]+)_seed([0-9]+)_'
DATASET_NAME=${BASH_REMATCH[2]}
EPOCHS=${BASH_REMATCH[3]}
SEED=${BASH_REMATCH[4]}

get_task_name $DATASET_NAME

#liuyan 0910 20：20
get_num_examples $DATASET_NAME

for am in $ATTACK_METHOD
do
  logfile=experiment_result/upper_only_attack/${MODEL_DIR}_${am}.log
  nohup python upper_attack.py \
      --dataset_name $DATASET_NAME \
      --task_name $TASK_NAME \
      --epochs $EPOCHS \
      --cuda $CUDA \
      --modeldir $MODEL_DIR \
      --seed $SEED \
      --attack_method $am \
      --num_examples $NUM_EXAMPLES \
      > $logfile 2>&1
done
