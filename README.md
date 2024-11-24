# Synthesis Pyramid Pooling: A Strong Pooling Method for Gait Recognition in the Wild
[Paper](https://doi.org/10.1109/LSP.2024.3470749) has been accepted in IEEE Signal Processing Letters. This is the code for it.
# Operating Environments
## Pytorch Environment
* Pytorch=1.10.1
# CheckPoints
* The checkpoint for Gait3D [link]().
* The checkpoint for GREW [link]().
# Train and Test
## Train
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1354 opengait/main.py --cfgs ./configs/SPP/gait3d_spp.yaml --phase train
```
## Test
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1354 opengait/main.py --cfgs ./configs/SPP/gait3d_spp.yaml --phase test
```
* python -m torch.distributed.launch: DDP launch instruction.
* --nproc_per_node: The number of gpus to use, and it must equal the length of CUDA_VISIBLE_DEVICES.
* --cfgs: The path to config file.
* --phase: Specified as train or test.
# Acknowledge
The codebase is based on [OpenGait](https://github.com/ShiqiYu/OpenGait).
# Citation
```
@article{peng2024synthesis,
  title={Synthesis Pyramid Pooling: A Strong Pooling Method for Gait Recognition in the Wild},
  author={Peng, Guozhen and Li, Rui and Li, Annan and Wang, Yunhong},
  journal={IEEE Signal Processing Letters},
  year={2024},
  publisher={IEEE}
}
```
