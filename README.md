# DL

*IML-CompAI DL*
Overview Deep Learning Framework

![Framework overview](./iml_dl.png)


Installation guide: 

# UNZIP
1). clone iml-dl to desired location (TARGET_DIR)

# Install required packages 
3). create a virtual environment with the needed packages 
	conda create env conda_requirements.yaml
	source activate iml_py308 
4). install pytorch and cuda
	pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

5). go to directory and run one script: 
	cd ${TARGET_DIR}/iml-dl
	python core/Main.py --config.path projects/dummy_project/config_cifar10.yaml 
	
# That's it!  
