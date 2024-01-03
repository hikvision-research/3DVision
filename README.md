# 3DVision

This is the opensourced 3D Vision repository of Hikvision Research Institute, China.

## Implementations

To date, 3DVision contains the following algorithms:

### Point Cloud Completion

[FBNet: Feedback Network for Point Cloud Completion. ECCV, 2022. Oral](./PointCompletion/FBNet)

### Point Cloud Upsampling
[Point Cloud Upsampling via Cascaded Refinement Network. ACCV, 2022. Oral](./PointUpsampling/PUCRN)

[Arbitrary-Scale Point Cloud Upsampling by Voxel-based Network with Latent Geometric-Consistent Learning. AAAI, 2024](./PointUpsampling/PU-VoxelNet)

### Point Cloud Normal Estimation
[Rethinking the Approximation Error in 3D Surface Fitting for Point Cloud Normal Estimation. CVPR, 2023](./Normal-Estimation)

### Sparse-View Reconstruction
[1st Place Solution in OmniObject3D Challenge Track-1. ICCV, 2023](./SparseReconstruction/solution)


## License

This project is released under the [Apache 2.0 license](./LICENSE). Other codes from open source repository follows the original distributive licenses.

## Acknowledgement

Some code of this repository is borrowed from [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet), [pytorchpointnet++](https://github.com/erikwijmans/Pointnet2_PyTorch), [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch),[MVP_Benchmark](https://github.com/paul007pl/MVP_Benchmark). We thank the authors for their great jobs!
