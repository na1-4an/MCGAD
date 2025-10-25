python run.py --dataset yelpchi    --alpha 0.1 --lr 0.0001 --weight_decay 0 --beta 0.1
python run.py --dataset book       --alpha 0.2 --lr 0.0003 --weight_decay 0 --bn True --beta 0.3
python run.py --dataset questions  --alpha 0.2 --lr 0.0005 --weight_decay 0 --num_epoch 500 --beta 0.1
python run.py --dataset Amazon-all --alpha 0.1 --lr 0.0003 --weight_decay 0.0005 --beta 0.4
python run.py --dataset reddit     --alpha 0.4 --lr 0.0001 --weight_decay 0 --bn True --beta 0.1
python run.py --dataset tolokers   --alpha 0.9 --lr 0.0001 --weight_decay 0.0005 --bn True --beta 0.7
python run.py --dataset photo      --alpha 0.9 --lr 0.0005 --weight_decay 0  --beta 0.7
python run.py --dataset elliptic   --alpha 0.9 --lr 0.0005 --weight_decay 0 --num_epoch 500 --beta 1.1
python run.py --dataset t_finance  --alpha 0.7 --lr 0.0005 --weight_decay 0.0005 --num_epoch 500 --bn True --beta 1.1