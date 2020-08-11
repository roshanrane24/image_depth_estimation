# image_depth_estimation
Image depth estimation using neural network.

Usage
=
```commandline
python main.py -i path/to/input [./input] -o path/to/output [./output] -m path/to/model [./model.pt]
```

Requirement
=
```buildoutcfg
numpy==1.19.1
opencv-python==4.3.0.36
torch==1.6.0+cpu
torchvision==0.7.0+cpu
```
Citation
=
```
@article{Ranftl2019,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {arXiv:1907.01341},
	year      = {2019},
}
```
### Based on
[MiDas | intel-isl](https://github.com/intel-isl/MiDaS)