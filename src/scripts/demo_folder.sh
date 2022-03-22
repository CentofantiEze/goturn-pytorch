MODEL_DIR='../goturn/models/caffenet/lightning_logs/version_0/'
FOLDER='../../test/bag_jpg/'
MASKS='../../test/bag_png/'

python demo_folder.py --model_dir $MODEL_DIR\
	--input $FOLDER \
	--ground_truth $MASKS \
