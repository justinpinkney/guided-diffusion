image_name = guided-diffusion
build:
	docker build -t $(image_name) .

datadir = /data/ffhq/images1024x1024
resultsdir = $(shell pwd)/results

run:
	docker run --gpus all -it --rm \
	--ipc=host --net=host --user $(id -u):$(id -g) \
	-v $(PWD):/workspace/ \
	-v $(datadir):/data \
	-v $(resultsdir):/results \
	$(image_name)

MODEL_FLAGS=--attention_resolutions 32,16,8 --diffusion_steps 1000 \
	--large_size 256 --small_size 128 --learn_sigma True --noise_schedule linear \
	--num_channels 192 --num_head_channels 64 --num_res_blocks 2 \
	--resblock_updown True --use_fp16 True --use_scale_shift_norm True
TRAIN_FLAGS=--lr 1e-4 --batch_size 8
export OPENAI_LOGDIR=/results/$(shell date +%Y%m%d-%H%M%S)
train-super:
	mpiexec -n 2 -mca pml ob1 -mca btl ^openib --allow-run-as-root \
	python scripts/super_res_train.py \
	--data_dir /data \
	$(MODEL_FLAGS) $(TRAIN_FLAGS)

train-im:
	mpiexec -n 2 -mca pml ob1 -mca btl ^openib --allow-run-as-root python scripts/image_train.py --data_dir /data