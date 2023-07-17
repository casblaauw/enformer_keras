# Sonnet-less Enformer
A Keras implementation of DeepMind's Enformer model. Largely based on Esteban Antonicelli's [earlier work](https://github.com/geantonicelli/enformer), to which I am very grateful. This implementation is updated to align more with Keras best practices and to enable use with [fastISM](https://github.com/kundajelab/fastISM).

To keep using the original ModelConfig setup, look at the `hconfig` branch. However, this is an outdated, untested branch, so keep that in mind.

To port weights from the pytorch implementation of Enformer to Keras, have a look at Esteban's original [implementation of that porting](https://github.com/geantonicelli/enformer/blob/main/torch_to_keras.py) and adapt it to the new structure as described in [snt_to_keras.py](snt_to_keras.py).