# img2ImgGAN
Implementation of the paper : "Toward Multimodal Image-to-Image Translation"

- Link to the paper : [ArXiv:1711.11586](https://arxiv.org/abs/1711.11586)
- Original PyTorch implementation: [Link]()
- Summary of the paper: [Github Gist]()

## Dependencies

- tensorflow (1.4.0)

## Hierarchy

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

## Generating the datasets

- Download the datasets from the following links
   - [edges2handbags]()
   - [edges2shoes]()
   - [facades]()
   - [maps]()

- To generate numpy files for the datasets,
   ```bash
   $ python main.py --create <dataset_name>
   ```

   This creates `train.npy` and `val.npy` in the corresponding dataset directory

- Alternate to the above step, you could read the images in real time during
  training. To do this, you should create files containing paths to the images.
  This can be done by running the following script in the root of this repo.
  ```bash
  $ bash setup_dataset.sh
  ```

## Running

### Generating graph:

To visualize the connections between the graph nodes, we can
generate the graph using the flag `archi`. This would be useful to assert the connections are correct.
```bash
$ python main.py --model <model_name> --archi
```
Eg:
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

### Complete list of options:

```bash
$ python main.py --help
```
