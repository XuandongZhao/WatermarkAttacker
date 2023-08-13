# WatermarkAttacker

This repository contains the code for the paper [Invisible Image Watermarks Are Provably Removable Using Generative AI](https://arxiv.org/abs/2306.01953).

We propose a family of **regeneration attacks** to remove invisible image watermarks. The attack method effectively removes invisible watermarks. 

Our attack first maps the watermarked image to its embedding, which is another representation of the image. Then the embedding is noised to destruct the watermark. After that, a regeneration algorithm reconstructs the image from the noisy embedding. As shown in the figure below:

![demo](./fig/demo.png)



<!-- If you find this repository useful, please cite our paper:

```
@article{zhao2023provable,
  title={Provable Robust Watermarking for AI-Generated Text},
  author={Zhao, Xuandong and Ananth, Prabhanjan and Li, Lei and Wang, Yu-Xiang},
  journal={arXiv preprint arXiv:2306.17439},
  year={2023}
}
``` -->


## Example

First, install the dependencies:

```bash
pip install -r requirements.txt
```

Then, run the following command to install modified [diffusers](https://github.com/huggingface/diffusers)

```bash
pip install -e .
```

Then you can try the demo in `demo.ipynb`.