conda env create -f ./environment.yml

sudo docker exec -it fth_simai05 bash

conda activate fth-pytorch-gpu-env

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers -i  https://mirrors.aliyun.com/pypi/simple/ 

pip install accelerate -i  https://mirrors.aliyun.com/pypi/simple/ 

pip install -U bitsandbytes

pip install -U bitsandbytes -i https://mirrors.aliyun.com/pypi/simple/ 

pip install torchsummary -i https://mirrors.aliyun.com/pypi/simple/ 
pip install matplotlib -i https://mirrors.aliyun.com/pypi/simple/ 

pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/ 

export MODELSCOPE_CACHE=/disk2/modelscope

modelscope download --model AI-ModelScope/bert-base-uncased



# 检查是否已存在该变量
grep 'MODELSCOPE_CACHE' ~/.bashrc

# 如果不存在，则写入
echo 'export MODELSCOPE_CACHE=/disk2/modelscope' >> ~/.bashrc

# 使配置生效
source ~/.bashrc

# 验证变量是否生效
echo $MODELSCOPE_CACHE