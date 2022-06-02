# Acoustic (Audio) Scene Classification
PyTorch Implementation of Deep Neural Network Training | DCASE2020 10 Class Scene Classification Dataset

### Execute the below codes

#### Dataset Downloading and Extraction

```
git clone https://github.com/akhilkumardonka/RobotAudition.git
cd ./RobotAudition/Audio\ Scene\ Classification/datasets
python downloader.py
python extracter.py
```

#### Directory Structure

    .
    ├── datasets
    │   ├── TAU-urban-acoustic-scenes-2020-mobile-development
    │       ├── audio                                                 # Contains all wav files
    │       ├── meta.csv
    │       ├── evaluation_setup                                      # Contains train/validation split
    │           ├── fold1_evaluate.csv
    │           ├── fold1_train.csv
    ├── training                    
    

### References

| Source | Link |
| ------ | ------ |
| Audio Classification using Librosa and Pytorch | https://medium.com/@hasithsura/audio-classification-d37a82d6715 |
| PyTorch Official Documentation | https://pytorch.org/tutorials/ |
