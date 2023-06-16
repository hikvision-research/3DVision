# Rethinking the Approximation Error in 3D Surface Fitting for Point Cloud Normal Estimation (CVPR 2023)

## Introduction

This is the official Pytorch implementation of our paper Rethinking the Approximation Error in 3D Surface Fitting for Point Cloud Normal Estimation (CVPR 2023). 

## Data Preparation

We use two public datasets, including PCPNet ([Paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13343), [Download](http://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/pclouds.zip)), and SceneNN ([Paper](https://ieeexplore.ieee.org/document/7785081/;jsessionid=scF2IPOwVm_zXH316Z4vskCszymli57XMe5zyYdfqEH6y4Pz825L!886754278), [Download](https://drive.google.com/drive/folders/0B-aa7y5Ox4eZWE8yMkRkNkU4Tk0)). 



## Usage

The code is tested under Pytorch 1.6.0 and Python 3.6 on Ubuntu 16.04. Pretrained models are available in ` ./log/ `. 

1. Install python denpendencies.
```shell
pip install -r requirements.txt
```


2. Compile pyTorch extensions.

```shell
cd pointnet2_ops_lib
python setup.py install

```

3. Train the model. 

```shell
sh start_train.sh
```

4. Evaluate the model.

```shell
sh test_all.sh
```

## License

This project is released under the [Apache 2.0 license](./LICENSE). Other codes from open source repository follows the original distributive licenses.


## Acknowledgement

We borrow some code from [DeepFit](https://github.com/sitzikbs/DeepFit), [AdaFit](https://github.com/Runsong123/AdaFit), and [GraphFit](https://github.com/UestcJay/GraphFit). Thanks for their excellent jobs.


## Citation
Please consider citing the follow papers in your publications if the project helps your research. 


```BibTeX
@InProceedings{Du_2023_CVPR,
    author    = {Du, Hang and Yan, Xuejun and Wang, Jingjing and Xie, Di and Pu, Shiliang},
    title     = {Rethinking the Approximation Error in 3D Surface Fitting for Point Cloud Normal Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9486-9495}
}

@inproceedings{li2022graphfit,
    title={GraphFit: Learning Multi-scale Graph-Convolutional Representation for Point Cloud Normal Estimation},
    author={Li, Keqiang and Zhao, Mingyang and Wu, Huaiyu and Yan, Dong-Ming and Shen, Zhen and Wang, Fei-Yue and Xiong, Gang},
    booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXII},
    pages={651--667},
    year={2022},
    organization={Springer}
}

@inproceedings{zhu2021adafit,
    title={AdaFit: Rethinking Learning-based Normal Estimation on Point Clouds},
    author={Zhu, Runsong and Liu, Yuan and Dong, Zhen and Wang, Yuan and Jiang, Tengping and Wang, Wenping and Yang, Bisheng},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={6118--6127},
    year={2021}
}

@inproceedings{ben2020deepfit,
    title={Deepfit: 3d surface fitting via neural network weighted least squares},
    author={Ben-Shabat, Yizhak and Gould, Stephen},
    booktitle={European conference on computer vision},
    pages={20--34},
    year={2020},
    organization={Springer}
}
```
