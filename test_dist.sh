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

oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

#$Config

NNODES=${NNODES:-1}

HOSTFILE=${HOSTFILE:-./hostfile}

BATCH_SIZE=3264

python -m intel_extension_for_pytorch.cpu.launch --distributed --nnodes ${NNODES} --hostfile ${HOSTFILE} train_test.py -a $ARGS --epochs 1 --warmup-epochs 2  --ipex -j 0 -b $BATCH_SIZE --seed 2020 \
        --dist-backend=ccl --base-op=LARS --base-lr 10.5 --weight-decay 0.00005 2>&1 | tee log_$3.log

sleep 5

