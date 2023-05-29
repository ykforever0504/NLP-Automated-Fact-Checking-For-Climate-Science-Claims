## Requirements

- python 3.8
- [PyTorch](https://pytorch.org/) >=1.10.0
- [Transformers](https://huggingface.co/transformers/) >=4.8.1


```angular2html
pip install -r requirements.txt
```

## Directory and File introduction
```angular2html
-dpr(task1: look for evidences related to the claim)
  -config.py(dpr model parameters setting)        
  -data_utils.py(input data processing)
  -generate_negative_samples.py(negative_sampling)
  -main.py(train/test model)
-cls(task2: determine the claim label)
  -config.py(cls model parameters setting) 
  -data_utils.py(input data processing)
  -main.py(train/test model)
  -model.py(improved classification model)
```
## Train and test

### DPR
```angular2html
train task1: nohup python -u dpr/main.py >train.out 2>&1 &
test task1: nohup python -u dpr/main.py -p --model_pt dpr >test_task1.out 2>&1 &
```

### CLS
```angular2html
train task2: nohup python -u cls/main.py >train.out 2>&1 &
test task2: nohup python -u cls/main.py -p --model_pt cls >test.out 2>&1 &
```
### Test dev
```angular2html
python eval.py --predictions dev-claims-predictions.json --groundtruth data/dev-claims.json
```
## Copy code website
```angular2html
pytorch official website: https://pytorch.org/tutorials/beginner/
Hugginface: https://huggingface.co/docs
```

