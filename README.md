# img2ImgGAN
Implementation of the paper : **Toward Multimodal Image-to-Image Translation**

- Link to the paper : [arXiv:1711.11586](https://arxiv.org/abs/1711.11586)
- PyTorch implementation: [Link](https://github.com/junyanz/BicycleGAN)
- Summary of the paper: [Gist](short_notes.md)


# Results
<div class="fig figcenter fighighlight">
  <img src="imgs/g_0.png" width="23%" style="margin-right:3px;"> 
  <img src="imgs/t_0.png" width="23%" style="margin-right:2px;">
  <img src="imgs/cLR_0.png" width="23%" style="margin-right:3px;">
  <img src="imgs/cVAE_0.png" width="23%">
</div>
<div class="fig figcenter fighighlight">
  <img src="imgs/g_1.png" width="23%" style="margin-right:3px;"> 
  <img src="imgs/t_1.png" width="23%" style="margin-right:2px;">
  <img src="imgs/cLR_1.png" width="23%" style="margin-right:3px;">
  <img src="imgs/cVAE_1.png" width="23%">
</div>
<div class="fig figcenter fighighlight">
  <img src="imgs/g_2.png" width="23%" style="margin-right:3px;"> 
  <img src="imgs/t_2.png" width="23%" style="margin-right:2px;">
  <img src="imgs/cLR_2.png" width="23%" style="margin-right:3px;">
  <img src="imgs/cVAE_2.png" width="23%">
</div>
<div class="fig figcenter fighighlight">
  <img src="imgs/g_3.png" width="23%" style="margin-right:3px;"> 
  <img src="imgs/t_3.png" width="23%" style="margin-right:2px;">
  <img src="imgs/cLR_3.png" width="23%" style="margin-right:3px;">
  <img src="imgs/cVAE_3.png" width="23%">
</div>
<div class="fig figcenter fighighlight">
  <img src="imgs/g_4.png" width="23%" style="margin-right:3px;"> 
  <img src="imgs/t_4.png" width="23%" style="margin-right:2px;">
  <img src="imgs/cLR_4.png" width="23%" style="margin-right:3px;">
  <img src="imgs/cVAE_4.png" width="23%">
</div>
<div class="fig figcenter fighighlight">
  <img src="imgs/g_5.png" width="23%" style="margin-right:3px;"> 
  <img src="imgs/t_5.png" width="23%" style="margin-right:2px;">
  <img src="imgs/cLR_5.png" width="23%" style="margin-right:3px;">
  <img src="imgs/cVAE_5.png" width="23%">

  <div class="figcaption">
    <b>Result</b> First column represents input, second column the ground truth. The next is the image generated from <b>cLR-GAN</b> and the last column represents the image generated from <b>cVAE-GAN</b>. Results were obtained from validation dataset.
  </div>
</div>


# Contents:
  - [Tensorboard Visualization of models](#tensorboard)
    - [cVAE-GAN](#cvae)
    - [cLR-GAN](#clr)
    - [Bicycle-GAN](#bicycle)
  - [Dependencies](#depends)
  - [Structure](#struct)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Generate Graph](#graph)
    - [Training the network](#train)
    - [Testing the network](#test)
  - [Visualizations](#viz)
  - [TODO](#todo)
  - [Licence](#licence)


<a name='tensorboard'></a>
# Model Architecture Visualization

- Network

![](imgs/network.jpeg)
**Fig 1:** Structure of BicycleGAN. (Image taken from the paper)

- Tensorboard visualization of the entire network

<a name='cvae'></a>
### cVAE-GAN Network
![](imgs/cvae-gan.gif)

<a name='depends'></a>
# Dependencies

- tensorflow (1.4.0)
- numpy (1.13.3)
- scikit-image (0.13.1)
- scipy (1.0.0)

To install the above dependencies, run:

```bash
$ sudo pip install -r requirements.txt
```

<a name='struct'></a>
# Structure

```
 -img2imgGAN/
            -nnet
            -utils
            -data/
                  -edges2handbags
                  -edges2shoes
                  -facades
                  -maps
```

<a name='setup'></a>
# Setup

- Download the datasets from the following links
   - [edges2handbags]()
   - [edges2shoes]()
   - [facades]()
   - [maps]()

- To generate numpy files for the datasets,
   ```bash
   $ python main.py --create <dataset_name>
   ```

   This creates `train.npy` and `val.npy` in the corresponding dataset directory. This generates very huge files. As an
   alternate, the next step attempts to read images at run-time during training 

- Alternate to the above step, you could read the images in real time during
  training. To do this, you should create files containing paths to the images.
  This can be done by running the following script in the root of this repo.
  ```bash
  $ bash setup_dataset.sh
  ```

<a name='usage'></a>
# Usage

<a name='graph'></a>
- Generating graph:

  To visualize the connections between the graph nodes, we can
  generate the graph using the flag `archi`. This would be useful to assert the connections are correct.
  This generates the graph for `bicycleGAN`
  ```bash
  $ python main.py --archi
  ```
  To generate the model graph for `cvae-gan`,
  ```bash
  $ python main.py --model cvae-gan --archi
  ```
  Possible models are:
  `cvae-gan`, `clr-gan`, `bicycle` (default)

  To visualize the graph on `tensorboard`, run the following command:
  ```bash
  $ tensorboard --logdir=logs/summary/Run_1 --host=127.0.0.1
  ```
  Replace `Run_1` with the latest directory name

- Complete list of options:

  ```bash
  $ python main.py --help
  ```

<a name='train'></a>
- Training the network

  To train `model` (say `cvae-gan`) on `dataset` (say `facades`) from scratch,
  ```bash
  $ python main.py --train --model cvae-gan --dataset facades
  ```

  The above command by default trains the model in which images from distribution of `domain B` are generated
  conditioned on the images from the distribution of `domain A`. To switch the direction,
  ```bash
  $ python main.py --train --model cvae-gan --dataset facades --direction b2a
  ```

  To resume the training from a checkpoint,
  ```bash
  $ python main.py --resume <path_to_checkpoint> --model cvae-gan
  ```

<a name='test'></a>
- Testing the network

  - Download the checkpoint file from [here](https://drive.google.com/drive/folders/1rXgCU-HS7AnlTiNzKRl-eUL2qsWhrGs8?usp=sharing) and place the checkpoint files in the `ckpt` directory
  
  To test the model from the given trained models,
  ```bash
  $ ./test.sh <dataset_name> <test_image_path> <checkpoint_path>
  ```

<a name='viz'></a>
# Visualizations
Loss of discriminator and generator as function of iterations on `edges2shoes` dataset.

![](imgs/loss/d_fake.png)
![](imgs/loss/d_real.png)
![](imgs/loss/d_loss.png)
![](imgs/loss/g_loss.png)


<a name='todo'></a>
# TODO
- [x] Residual Encoder
- [ ] Multiple discriminators for `cVAE-GAN` and `cLR-GAN`
- [ ] Inducing noise to all the layers of the generator
- [ ] Train the model on rest of the datasets

<a name='licence'></a>
# License

Released under [the MIT license](LICENSE)
