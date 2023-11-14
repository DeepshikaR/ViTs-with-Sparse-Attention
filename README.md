# SparseAttentionViT


## Dataset

[Imagenett2](https://github.com/fastai/imagenette)

Due to GPU and data constraints

## Experiment

Metric: Classfication Accuracy
Loss: CrossEntropy
Optimizer: Adam

### Configurations

Original ViT:
ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)



## Results

| Model             | Accuracy    | 
| -----------       | ----------- |
| Original ViT      |    75,66 small,64 small p8, 66 small p8 mean    |
| Adapted BigBird           | 73.85,     61 small p8 mean     |
| Random Attention           | 68 small , 65 small p8. 58 small p8 mean       |
| Random Attention + Global          | - , -, 60 small p8 mean       |
| Random Attention + Window          | - , -, 61.7 small p8 mean       |
| Windowed Attention           | -           | 63.6
| Windowed Attention   + Global        | -           | 62
| Global Attention           | -           |


Combinations of the above




| Model             | Accuracy    | 
| -----------       | ----------- |
| Original ViT                |53.33|
| Adapted BigBird             |64.58|
| Random Attention            |59.37|
| Random Attention + Global   |62.29|
| Random Attention + Window   |63.39|
| Windowed Attention          |66.25| 
| Windowed Attention+ Global  |65.66|
| Global Attention            |62.39|


