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
python train.py --num-workers 0 --gradient-accumulation-steps 1 --eval-every 5
```