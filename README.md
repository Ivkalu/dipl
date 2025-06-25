# Masters thesis

🌱 Setup environment
```
python3 -m venv venv
source venv/bin/activate //linux
./venv//activate.bat //windows
pip install -r requirements.txt
```
⏳ Download the data
```
python src/
```

🧬 Project structure
```
.
├── ...
├── data
│   ├── model_output
│   ├── Test_submission
│   ├── test_y
│   │   ├── distortion
│   │   └── reverb
│   ├── Train submission
│   ├── test_y
│   │   ├── distortion
│   │   └── reverb
│   ├── Metadata_Test.csv
│   └── Metadata_Train.csv
├── models
├── research
├── src
└── ...
```

denotes to hidden folders


## Models

Pedals:
- reverb
- distortion
- delay


Regular:
- Baseline
- LSTM - RNN
- Wavenet
- T4
- Transformer
- U-net
- (Mamba)

Paramtric based models
Genetic algorithm:
- spotify - pedal (probably shit)

Gradient based:
- DASP (it will learn faster than genetic, it's parameter space is same as genetic, that means it is limited in expressivity, although it should be "good enough" for approximating most pedals, and should learn a lot faster than regular models)

For best model:
- try out dataset, no guitar in  trainset, but only in valid, compare results
- try training with different LOSS functions (mse, spectrogram loss, ...)


