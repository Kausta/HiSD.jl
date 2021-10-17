[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](LICENSE.md)

# HiSD: Image-to-image Translation via Hierarchical Style Disentanglement

Unofficial Knet.jl implementation of paper "[Image-to-image Translation via Hierarchical Style Disentanglement](https://arxiv.org/abs/2103.01456)".

This version was implemented by Caner Korkmaz for the Koç University Comp 541 Course and ML Reproducibility Challenge 2021 Fall Edition. You can find the original implementation in [https://github.com/imlixinyang/HiSD](https://github.com/imlixinyang/HiSD).

## License

Following the licensing of the original implementation, this code is licensed under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International).

The code in this repo was created from scratch, using the previously cited original implementation and the paper as references.

The code is released for academic research use only following the original implementation. For other use, please check the original implementation or contact the author of the official pytorch implementation at [imlixinyang@gmail.com](mailto:imlixinyang@gmail.com).

## Citation

If the original paper helps your research, please cite it in your publications:
```
@InProceedings{Li_2021_CVPR,
    author    = {Li, Xinyang and Zhang, Shengchuan and Hu, Jie and Cao, Liujuan and Hong, Xiaopeng and Mao, Xudong and Huang, Feiyue and Wu, Yongjian and Ji, Rongrong},
    title     = {Image-to-Image Translation via Hierarchical Style Disentanglement},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {8639-8648}
}
```


## Related Work

- Multi-style/modal: [MUNIT](https://github.com/NVlabs/MUNIT), [DRIT](https://github.com/HsinYingLee/DRIT), [ContentDisentanglement](https://github.com/oripress/ContentDisentanglement), etc.
- Multi-label: [StarGAN](https://github.com/yunjey/stargan), [STGAN](https://github.com/csmliu/STGAN), [RelGAN](https://github.com/elvisyjlin/RelGAN-PyTorch), etc.
- Joint: [SMIT](https://github.com/BCV-Uniandes/SMIT), [SDIT](https://github.com/yaxingwang/SDIT), [DMIT](https://github.com/Xiaoming-Yu/DMIT), [AGUIT](https://github.com/imlixinyang/AGUIT), [ELEGANT](https://github.com/Prinsphield/ELEGANT), [StarGANv2](https://github.com/clovaai/stargan-v2), etc.