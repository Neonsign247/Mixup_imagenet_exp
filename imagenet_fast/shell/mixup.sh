GPU=0,1,2,3,4,5,6,7
NAME=mixup

DATA160=/workspace/dataset/ILSVRC2012-sz/160
DATA352=/workspace/dataset/ILSVRC2012-sz/352

CONFIG1=configs/${NAME}/configs_fast_phase1.yml
CONFIG2=configs/${NAME}/configs_fast_phase2.yml
CONFIG3=configs/${NAME}/configs_fast_phase3.yml

OUT1=out/${NAME}_train_phase1.out
OUT2=out/${NAME}_train_phase2.out
OUT3=out/${NAME}_train_phase3.out

EVAL1=eval/${NAME}_eval_phase1.out
EVAL2=eval/${NAME}_eval_phase2.out
EVAL3=eval/${NAME}_eval_phase3.out

for i in 5
do
    PREFIX1=${NAME}_${i}_phase1
    PREFIX2=${NAME}_${i}_phase2
    PREFIX3=${NAME}_${i}_phase3

    END1=./trained_models/${PREFIX1}/checkpoint_epoch15.pth.tar
    END2=./trained_models/${PREFIX2}/checkpoint_epoch40.pth.tar
    END3=./trained_models/${PREFIX3}/checkpoint_epoch100.pth.tar
    # training for phase 1
    CUDA_VISIBLE_DEVICES=$GPU python -u main_fast_mixup.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 | tee $OUT1

    # evaluation for phase 1
    # CUDA_VISIBLE_DEVICES=$GPU python -u main_fast.py $DATA352 -c $CONFIG1 --output_prefix $PREFIX1 --resume $END1  --evaluate | tee $EVAL1

    # training for phase 2
    CUDA_VISIBLE_DEVICES=$GPU python -u main_fast_mixup.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END1 | tee $OUT2

    # evaluation for phase 2
    # CUDA_VISIBLE_DEVICES=$GPU python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END2 --evaluate | tee $EVAL2

    # training for phase 3
    CUDA_VISIBLE_DEVICES=$GPU python -u main_fast_mixup.py $DATA352 -c $CONFIG3 --output_prefix $PREFIX3 --resume $END2 | tee $OUT3

    # evaluation for phase 3
    # CUDA_VISIBLE_DEVICES=$GPU python -u main_fast.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END3 --evaluate | tee $EVAL3

done
