# img2imgGAN
Implementation of the paper : "Toward Multimodal Image-to-Image Translation"

- Link to the paper : [ArXiv:1711.11586](https://arxiv.org/abs/1711.11586)

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

## Running

### For the complete list of options run,

```bash
$ python main.py --help
```
