# Create Env
```
conda create -n project python=3.7 pytorch=1.3 torchvision matplotlib imageio
```

or
```
conda install -c pytorch torchvision
conda install -c conda-forge matplotlib
conda install -c menpo imageio
conda install -c effdet
```

The (in either case):
```
pip install timm
```

from within `Visual C++ 2015 x86 x64 Cross Build Tools Command Prompt` cd to the project folder, then run:
```
activate project
pip install pycocotools
```

then back to Miniconda:
```
activate project
pip install effdet
pip install pandas
conda install -c conda-forge opencv
```

# Test Model
in Miniconda:
```
activate project
python Main.py -myAnns newAnns.txt -anns annotationsTrain.txt -buses busesDir -saveDir dirToSave
```

# Train
in Miniconda, run:
```
activate project
cd Code
python train.py --num-workers 2 --gradient-accumulation-steps 2 --eval-every 10
```

# Google Colab Integration
Add the following code
```
from google.colab import drive
drive.mount('/content/drive/')
%cd '/content/drive/Shareddrives/Computer Vision Project/Code/'

if True:
    # the next command will install both pytorch and torchvision
    !pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    # install all other packages
    packages = ['matplotlib', 'imageio', 'effdet', 'timm', 'pycocotools', 'pandas', 'opencv-python']
    for package in packages:
        !echo $package
        !pip install $package
```

Then call train:
```
!python train.py --num-workers 4 --gradient-accumulation-steps 2 --eval-every 10 --batch-size 6 --log-to-file false
```

80AP train:
```
!python train.py --num-workers 4 --gradient-accumulation-steps 2 --eval-every 50 --batch-size 2 --log-to-file false --load-model false --num-epochs 500 --model-name tf_efficientdet_d3 --target-dim 896 --lr 0.007
```