# Masters thesis

🌱 Setup environment
```
python3 -m venv venv
source venv/bin/activate //linux
./venv//activate.bat //windows
pip install -r requirements.txt
```

🧬 Project structure
```
.
├── ...
├── checkpoints
├── plugins
├── data 
│   ├── test
│   │   ├── x
│   │   │   ├── guitar
│   │   │   └── other
│   │   └── y 
│   │       ├── guitar
│   │       │   ├── effect1
│   │       │   └── ...
│   │       └── other
│   │           ├── effect1
│   │           └── ...
│   └── train
│       ├── x
│       │   ├── guitar
│       │   └── other
│       └── y 
│           ├── guitar
│           │   ├── effect1
│           │   └── ...
│           └── other
│               ├── effect1
│               └── ...
│   
├── research
├── src
└── ...
```


## Models

Pedals:
- reverb
- distortion
- delay


Regular:
- Multilayer Perceptron
- LSTM - RNN
- Wavenet
- TCN
- Structured SSM S4
- Transformer
- GAN ?

Paramtric based models
Genetic algorithm:
- spotify - pedal (probably shit)

Gradient based:
- DASP (it will learn faster than genetic, it's parameter space is same as genetic, that means it is limited in expressivity, although it should be "good enough" for approximating most pedals, and should learn a lot faster than regular models)

For best model:
- try out dataset, no guitar in  trainset, but only in valid, compare results
- try training with different LOSS functions (mse, spectrogram loss, ...)


