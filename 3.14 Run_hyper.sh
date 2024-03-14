python run_experiment.py --model NeuCF --rounds 2 --density 0.01 --dataset rt --epochs 300 --bs 256 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1 --check 0
python run_experiment.py --model NeuCF --rounds 2 --density 0.05 --dataset rt --epochs 300 --bs 256 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1 --check 0
python run_experiment.py --model NeuCF --rounds 2 --density 0.10 --dataset rt --epochs 300 --bs 256 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1 --check 0

python run_experiment.py --model Reloop2 --rounds 2 --density 0.01 --dataset rt --epochs 300 --bs 256 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_experiment.py --model Reloop2 --rounds 2 --density 0.05 --dataset rt --epochs 300 --bs 256 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_experiment.py --model Reloop2 --rounds 2 --density 0.10 --dataset rt --epochs 300 --bs 256 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

