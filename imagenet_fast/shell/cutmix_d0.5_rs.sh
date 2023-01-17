GPU=6,7
NAME=cutmix_d0.5

DATA160=/workspace/dataset/ILSVRC2012-sz/160
DATA352=/workspace/dataset/ILSVRC2012-sz/352

CONFIG1=configs/${NAME}/configs_fast_phase1.yml
CONFIG2=configs/${NAME}/configs_fast_phase2.yml
CONFIG3=configs/${NAME}/configs_fast_phase3.yml

PREFIX1=${NAME}_randomseed_phase1
PREFIX2=${NAME}_randomseed_phase2
PREFIX3=${NAME}_randomseed_phase3

OUT1=out/${NAME}_train_phase1.out
OUT2=out/${NAME}_train_phase2.out
OUT3=out/${NAME}_train_phase3.out

EVAL1=eval/${NAME}_eval_phase1.out
EVAL2=eval/${NAME}_eval_phase2.out
EVAL3=eval/${NAME}_eval_phase3.out

END1=./trained_models/${PREFIX1}/checkpoint_epoch15.pth.tar
END2=./trained_models/${PREFIX2}/checkpoint_epoch40.pth.tar
END3=./trained_models/${PREFIX3}/checkpoint_epoch100.pth.tar

# training for phase 1
CUDA_VISIBLE_DEVICES=$GPU python -u cutmix_drop_rs.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 | tee $OUT1

# evaluation for phase 1
# CUDA_VISIBLE_DEVICES=$GPU python -u main_fast.py $DATA352 -c $CONFIG1 --output_prefix $PREFIX1 --resume $END1  --evaluate | tee $EVAL1

# training for phase 2
CUDA_VISIBLE_DEVICES=$GPU python -u cutmix_drop_rs.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END1 | tee $OUT2

# evaluation for phase 2
# CUDA_VISIBLE_DEVICES=$GPU python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END2 --evaluate | tee $EVAL2

# training for phase 3
CUDA_VISIBLE_DEVICES=$GPU python -u cutmix_drop_rs.py $DATA352 -c $CONFIG3 --output_prefix $PREFIX3 --resume $END2 | tee $OUT3

# evaluation for phase 3
# CUDA_VISIBLE_DEVICES=$GPU python -u main_fast.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END3 --evaluate | tee $EVAL3

