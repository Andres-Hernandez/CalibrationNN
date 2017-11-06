# Model calibration with Neural Networks
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2812140

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2996930

## Dependencies

* [TensorFlow >=1.0](https://www.tensorflow.org/)
* [keras >=2.0](https://github.com/keras)

## Calibrate history

```
python swaption_test.py --calibrate-history --model=g2++
```

Command-line flags:

* `model`: Financial Model to use: hullwhite, g2++, g2++_local (for g2++ calibrated with local optimizer)

## Prepare training data

```
python swaption_test.py --training-data=250000 --model=g2++ --seed=111 --error-adjusted --history-start=0 --history-end=132
```

Command-line flags:

* `training-data`: Size of sample to produce
* `model`: Financial Model to use: hullwhite, g2++, g2++_local (for g2++ calibrated with local optimizer), g2++_vae (use Variational Autoencoder to sample)
* `seed`: If present, number is used to set seed of random number generator
* `error-adjusted`: If present, samples are adjusted with errors taken from observed results. For a model that does not fit data perfectly, error adjustment is important
* `history-start`: If present, sets the start date from which the sample will be trained. Value is the row of the available dates. Needs to be given in conjunction with history-end. 
* `history-end`: If present, sets the end date from which the sample will be trained. Value is the row of the available dates. Needs to be given in conjunction with history-start. 

## Train neural network to calibrate model

```
python swaption_test.py --run-fnn=0.5 --model=g2++ --seed=111 --sample-size=500000 --error-adjusted --history-start=0 --history-end=264 --epochs=750 --residual-cells=1 --lr=0.0001 --exponent=6 --layers=9 --dropout=0.2 --earlyStopPatience=41 --reduceLRFactor=0.5 --reduceLRPatience=10 --reduceLRMinLR=0.000005 --prefix="SWO GBP " --save --compare-history
```

Command-line flags:

* `run-fnn`: Size of sample to use (a training set could have been produced with a large set in order to compare how set size affects training). Needs to be a number in (0, 1])
* `model`: Financial Model to use: hullwhite, g2++, g2++_local (for g2++ calibrated with local optimizer), g2++_vae (use Variational Autoencoder to sample)
* `seed`: If present, number is used to set seed of random number generator
* `sample-size`: Size of sample in file. Used in building automatically the file name it will look for
* `error-adjusted`: If present, samples are adjusted with errors taken from observed results. Used in building automatically the file name it will look for
* `history-start`: If present, sets the start date from which the sample will be trained. Used in building automatically the file name it will look for
* `history-end`: If present, sets the end date from which the sample will be trained. Used in building automatically the file name it will look for
* `prefix`: String to preapped as prefix to model name. Affects name of file to which model could be saved. Default is empty
* `postfix`: String to apped as prefix to model name. Affects name of file to which model could be saved. Default is empty
* `save`: Save model to file, which can later be reloaded
* `compare-history`: Indicates that after training, the model should be used to compare the prediction with the observed history
* `load`: Name of file where model was saved. Model will be loaded, but not retrained. Intended to use to allow to compare-history at a later time

### Neural network parameters
* `epochs`: Number of epochs to train for. If not present, default is 200
* `residual-cells`: If present, sets the number of layers that will be skipped in building residual feedback, e.g. 1 means input of layer is added to it's ouput, 2 means that input of layer K is added to output of layer K+1. If not present, default is 1
* `lr`: Learning rate of optimizer (Nadam). If not present, default is 1e-3
* `exponent`: Number of neurons in layer will be 2**exponent. If not present default is 6 (so 64 neurons)
* `layers`: Number of hidden layers. If not present, default is 4
* `dropout`: Percentage of cells randomly dropped during each epoch of training so as to avoid overfitting by fitting an ensemble of models. If not present, default is 1/2
* `earlyStopPatience`: Number of epochs after which training should be stopped if the score of the validation set has not improved. Default is not to stop
* `reduceLRFactor`: Reduce learning rate by this factor if after a number of epochs, the score of the validation set has not improved. Default is not to reduce
* `reduceLRPatience`: Reduce learning rate by a factor if after this number of epochs, the score of the validation set has not improved. Default is not to reduce
* `reduceLRMinLR`: Minimum level to which to reduce the learning rate, in case it is being reduced. 

## Load GBP csv files into HDF5 file
```
python swaption_test.py --gbp-to-h5
```
