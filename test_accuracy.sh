# doing inference:
#  bash test_accuracy.sh resnet50 DATASET_DIR bf16(fp32)

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024


CONFIG_FILE=""

if [ "x$TRAINING_EPOCHS" == "x" ]; then
    EPOCHS=90
else
    EPOCHS=$TRAINING_EPOCHS
fi

ARGS=""
ARGS="$ARGS $1"
echo "###### running $1 model #######"
if [ "$1" == "resnext101_32x16d_swsl" ]; then
    ARGS="$ARGS --hub"
fi

ARGS="$ARGS $2"
echo "### dataset path: $2"

if [ "$3" == "bf16" ]; then
    ARGS="$ARGS --bf16"
    echo "### running bf16 datatype"
else
    echo "### running fp32 datatype"
fi

BATCH_SIZE=128
python -u train_oneccl.py -a $ARGS --ipex --jit -j 0 -b $BATCH_SIZE --seed 2020 --base-op=LARS \
        -e --resume model_best.pth.tar 2>&1 | tee log_$3.log

sleep 5
