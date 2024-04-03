# ERM-KTP
Code for our CVPR 2023 paper "**[ERM-KTP: Knowledge-level Machine Unlearning via Knowledge Transfer](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_ERM-KTP_Knowledge-Level_Machine_Unlearning_via_Knowledge_Transfer_CVPR_2023_paper.html)**". 

## Requirements
>- Platforms: Ubuntu: 18.04 Cuda: 11.7
>- Python: 3.8 Pytorch: 1.10.1

## Initialize
~~~
Edit the file parser_init.py
~~~

## Train ERM-CNNs
~~~
python train_ERM.py
~~~

## KTP
~~~
python KTP.py
~~~

## Test
~~~
python classAcc_valid.py
~~~

##
If you want to unlearn the specific class in your work, please modidy the following code in line 262 of dataLoder.py.
~~~
if data_train[i][1] >= 0 and data_train[i][1] < self.num_unlearn:
~~~

## Citation

Cite as below if you find this repository helpful:

~~~ 
@InProceedings{Lin_2023_CVPR,
    author    = {Lin, Shen and Zhang, Xiaoyu and Chen, Chenyang and Chen, Xiaofeng and Susilo, Willy},
    title     = {ERM-KTP: Knowledge-Level Machine Unlearning via Knowledge Transfer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20147-20155}
}
~~~
