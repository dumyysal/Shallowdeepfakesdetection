# Shallow- and Deep-fake Image Manipulation detection By ByeTorch for The Deepfake Forensics Challenge – IEEE CS Tunisia & IEEE YP Tunisia AG 

   ![image](https://github.com/dumyysal/ShallowdeepfakesdetectionEnit/assets/150078373/7aecf8f6-d3c5-4d6a-95b4-f6b95ff89494)

Our project was based on this research paper  [Shallow- and Deep-fake Image Manipulation Localization Using Deep Learning](Paper.pdf) | [Our proposed scientific research paper ](Paper_ENIT_SB.pdf)



## Datasets

### Deepfakes

To facilitate the training and testing of the model , you can download the dataset by accessing the following link:[here](https://www.dropbox.com/s/o5410tl5v4vxsth/ICNC2023-Deepfakes.tar.xz?dl=0).

### Shallowfakes

For the Shallowfake dataset utilized in our research paper, individual downloads are available through the following links:

- [CASIAv2](https://github.com/namtpham/casia2groundtruth)
     Revised dataset: https://bit.ly/2QazgkG
- [CASIAv1](https://github.com/namtpham/casia1groundtruth)
                   or here (https://bit.ly/3nXyYJw)
- [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)
- [COVERAGE](https://github.com/wenbihan/coverage)
- [NIST16](https://www.nist.gov/itl/iad/mig/open-media-forensics-challenge)
            Ensure you have an account here to proceed with downloading the dataset.






##  environment set up
 - Ensure that the PyTorch extension version is 1.8.0, and that CUDA 12.1.0 is installed when running the model

### Train/Val/Test Subsets

The way (file paths) of how we split the datasets into train/val/test subsets in find at the paths folder . In case you use conda environment use conda_paths.

The format of each line in these files is as the following. For authentic images, `/path/to/mask.png` and `/path/to/egde.png` are set to string `None`. We use digit `0` to represent authentic images, and `1` to represent manipulated images.

```
/path/to/image.png /path/to/mask.png /path/to/egde.png 0/1
```

## Usage

### Training

Run the following code to train the network.

For the option `--model`, to reproduce experiments in Table III of our paper:

- Use `mvssnet` for experiments 1/2/3;
- Use `upernet` for experiments 4/5/6;
- Use `ours` for experiments 7/8/9.

```
python -u train_torch.py --paths_file /path/to/train.txt --val_paths_file /path/to/val.txt --model {mvssnet, upernet, ours}
```

### Testing

Run the following code to evaluate the network.

Trained models for experiments in Table III of our paper can be found in the following links: [1](https://www.dropbox.com/s/jov5nsj47pyfv16/1.pth?dl=0) | [2](https://www.dropbox.com/s/w9eviamadmc0feh/2.pth?dl=0) | [3](https://www.dropbox.com/s/4pq92dmjzepi0uk/3.pth?dl=0) | [4](https://www.dropbox.com/s/i9eakxvww8vsbh7/4.pth?dl=0) | [5](https://www.dropbox.com/s/0jx8pxq1aksir18/5.pth?dl=0) | [6](https://www.dropbox.com/s/adsvglkcwv6ttnj/6.pth?dl=0) | [7](https://www.dropbox.com/s/nr81w432k9llztc/7.pth?dl=0) | [8](https://www.dropbox.com/s/g2n58undkom78tb/8.pth?dl=0) | [9](https://www.dropbox.com/s/zzk4eump5xfbqmz/9.pth?dl=0).
python -u evaluate.py --paths_file /pathd/test.txt --load_path /path/to/trained/model.path --model {mvssnet, upernet, ours}


### Output Results of The Model Mvssnet for experiments 1/2/3

      - Here we wanted to provide examples of the contents that may be found while testing the mvssnet model, including masks and other relevant files.Check The out folder

### Output Results of The pretrained Models below On New Data
![image](https://github.com/dumyysal/ShallowdeepfakesdetectionEnit/assets/150078373/3b905862-c955-4e32-a755-3e3fd26e6e5f)
  - a series of predicted masks from experiments 7 ( pretrained model7.pth) to 9 (pretrained model9.pth) on deepfake is presented .
  - It explains the success of the model's performance when confronted with new deep fake type data .

    
 ![image](https://github.com/dumyysal/ShallowdeepfakesdetectionEnit/assets/150078373/f7e425a6-c1e8-4b07-a8eb-cfa9807a5d64)
 - The output mask from Experiment 9 closely aligns with the ground truth image . 

