# IML-CompAI DL

Overview Deep Learning Framework

![Framework overview](./iml_dl.png)


# Installation guide: 

1). Clone iml-dl to desired location 
 * git clone https://github.com/iml-compai-munich/iml-dl.git *TARGET_DIR*

2). Create a virtual environment with the needed packages 
* cd ${TARGET_DIR}/iml-dl
* conda create env create -f conda_requirements.yaml
* source activate iml_py308 

3). Install pytorch and cuda
* pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

4). Run the demo script: 
* python core/Main.py --config.path projects/dummy_project/config_cifar10.yaml 
	
# That's it, enjoy!
