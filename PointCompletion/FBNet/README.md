# FBNet: Feedback Network for Point Cloud Completion

This repository contains the source code for the paper [FBNet: Feedback Network for Point Cloud Completion. ECCV, 2022. Oral]().

## Preparation


Our project is compatible with [MVP_Benchmark](https://github.com/paul007pl/MVP_Benchmark) project.
The recommended environment requirements can be found in [MVP_Benchmark](https://github.com/paul007pl/MVP_Benchmark)

- Linux with Python >= 3.6
- We use pytorch=1.6


## Usage

You can copy our project to the directory MVP_Benchmark/completion/ and run our demo as follows:

```shell
cd FBNet
sh run.sh
```

The pretrain model on MVP (2048) dataset is available at [FBNet_pre-trained](./model), which get 5.05 CD-T error on MVP testset.

Training code will be released soon.

## Citation

If you use FBNet in your research or wish to refer to the results published in the paper, please consider citing our paper.

```BibTeX
@inproceedings{Yan2022FBNet,
    title={FBNet: Feedback Network for Point Cloud Completion},
    author={Xuejun Yan, Hongyu Yan, Jingjing Wang, Hang Du, Zhihong Wu, Di Xie, Shiliang Pu, Li Lu.},
    booktitle={European Conference on Computer Vision},
    year={2022},
}
```

## License

This project is released under the [Apache 2.0 license](./LICENSE). Other codes from open source repository follows the original distributive licenses.

## Acknowledgement

Some code of this repository is borrowed from [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet), [pytorchpointnet++](https://github.com/erikwijmans/Pointnet2_PyTorch), [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch),[MVP_Benchmark](https://github.com/paul007pl/MVP_Benchmark). We thank the authors for their great jobs!
