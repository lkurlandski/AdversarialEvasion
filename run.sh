
device="cuda:1"
epochs="20"
batch_size="512"
pretrained="/home/lk3591/Documents/courses/CSEC-720/AdversarialEvasion/output/False/False/None/18.pth"

echo "--attack=FGSM"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=FGSM;
echo "--------------------------------------------------------------------------------\n";
echo "--attack=IGSM"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=IGSM;
echo "--------------------------------------------------------------------------------\n";
echo "--attack=PGD"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=PGD;
echo "--------------------------------------------------------------------------------\n";

echo "--attack=FGSM --pretrained"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=FGSM --pretrained=$pretrained;
echo "--------------------------------------------------------------------------------\n";
echo "--attack=IGSM --pretrained"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=IGSM --pretrained=$pretrained;
echo "--------------------------------------------------------------------------------\n";
echo "--attack=PGD --pretrained"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=PGD --pretrained=$pretrained;
echo "--------------------------------------------------------------------------------\n";

echo "--attack=FGSM --pretrained"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=FGSM --use_only_first_model;
echo "--------------------------------------------------------------------------------\n";
echo "--attack=IGSM --pretrained"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=IGSM --use_only_first_model;
echo "--------------------------------------------------------------------------------\n";
echo "--attack=PGD --pretrained"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=PGD --use_only_first_model;
echo "--------------------------------------------------------------------------------\n";

echo "--attack=FGSM --pretrained"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=FGSM --pretrained=$pretrained --use_only_first_model;
echo "--------------------------------------------------------------------------------\n";
echo "--attack=IGSM --pretrained"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=IGSM --pretrained=$pretrained --use_only_first_model;
echo "--------------------------------------------------------------------------------\n";
echo "--attack=PGD --pretrained"
python main.py --device=$device --epochs=$epochs --batch_size=$batch_size --attack=PGD --pretrained=$pretrained --use_only_first_model;
echo "--------------------------------------------------------------------------------\n";
