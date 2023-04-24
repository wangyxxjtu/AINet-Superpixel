# Run our code via docker
You can reproduce our results in docker, following these steps.

## section 1: build the docker environment
Make sure you have Docker installed on your computer, if you do not know how to install docker, you can refer to [this page](https://docs.docker.com/engine/install/ubuntu/) . Following steps will let you build the required contrainer

##### step 1: download the docker image ainet.tar from [here](https://pan.baidu.com/s/11C8N8g7BiIeCfYfhzxRXXw?pwd=e7m3) (code: e7m3). Two files are shared, ainet.tar is the environment package, superpixel.zip is the code and processed dataset. superpixel.zip need to be unzip, suppose you unzip the superpixel.zip in /home/name/superpixel: 
```bash
>> unzip superpixel.zip
```

##### step 2: load the docker image
``` bash
>> docker load --input ainet.tar
```

##### step 3: build a container, , then 
```bash
>> nvidia-docker run -itd --name ainet -v /home/name:/home/yaxiong -p 3316:22 ainet:v1
```
##### step 4: run the contrainer
```bash
>> nvidia-docker start -i ainet
```
After the above steps, you will enter the docker container.


## seciont 2: Run and reproduce the results in our paper
Next, we introduce how to run and reproduce our reuslts. Note that since we have included the BSDS500 dataset in the Docker image, you don't need to download or prepare the dataset separately. You can just run the following commands to train and evaluate the models.

##### step 1: once you enter, you will be directly to the code directory, then run the training by:
```console
root@local$>> sh train.sh
```

##### step 2: after training, run test:
```console
root@local$>> sh test.sh
```
##### step 3: go to the directory of superpixel evaluation benchmark and run the evaluation script first:
```console
root@local$>> cd /home/yaxiong/superpixel/superpixel-benchmark/examples/bash/
```
##### then, run
```console
root@local$>> bash superpixel_eval.sh
```

##### step 4: go to the plot curve directory
```console
root@local$>> cd /home/yaxiong/superpixel/AINet-Pytorch/eval_spixel/
```


##### step 5: summarze the results and run the code, first summarize the results:
```console
root@local$>> python copy_resCSV.py --src ../../AINet-Pytorch/eval/test_multiscale_enforce_connect/ --dst save
```
##### then, plot the curve, the BP-BP, ASA, CO figures will be saved in ./save directory
```console
root@local$>> python plot_benchmark_curve.py --path save/
```
 with the above steps, the curves will be ploted and saved.


