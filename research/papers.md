# Notes on papers

pretrained vs transfer learning vs fine-tuning
model distillation
Diffusion models – Used in advanced guitar amp modeling (like Tone Transfer).
GANs (e.g., HiFi-GAN or GANSynth) – Can be trained for instrument modeling.
Wave-U-Net (CNN-Based) 

## WaveNet

It is **autoregressive model**


To explore:
- Causal convolutional layers
- Dilated convolutional layers
- Residual skip connections

Problems:
- Receptive field in Speech generator: 300ms
- Receptive field in TTS: 240ms

autoregressive meaning
end to end models

A vocoder is a model that converts acoustic features (such as mel spectrograms) into a waveform, enabling speech synthesis.

wavenet is actually similar to vocoder => trying to make a voice from a voice representation

Better vocoders: HiFi-GAN or Parallel WaveGAN

## Map music2vec
gi
self supervized learning
Kao se spaja na posljednje layere transformera da mu smanji broj parametara idk?
Is used for:
- genre classification
- key detection
- emotion recognition

Jukebox is the SOTA

## Transformers in audio
(A GENERATIVE MODEL FOR RAW AUDIO USING TRANSFORMER ARCHITECTURES)

Pljačka
Samo su bubnuli transformere i ispalo je bolje od waveneta no shit

## Spleeter

Fora stvar, navodno radi skoro jednako dobro ko ostali SOTA audio splitter modeling

## Starcraft

imitation learning, rein-
forcement learning, and multi-agent learning
uses KL-Divergence
scatter connections,

## Deep speech

SOTA speech to text model
uses RNNs to extract letters and words from audio, then uses LLMs to fix any errors (because RNN doesn't neccesarily know the meaning of those words and might make nonsese words)
Uses smart parallelization of GPUs.
Adds synthetic noise to dataset (by superimposing real noise, and removing any outliers from fft analysis.)
Ask chatGPT to explain it better (What are the differences between cortana and siri...)

## NEAT

uses speciation as a core principal

## Shazam

koristi cacheiranje

## Fast speech

non-autoregressive

## Real-Time Guitar Amplifier Emulation with Deep Learning

aliasing??
MUSHRA listening tests
Istrenirani sa samo 3 minute audia!

### WaveNet Experiment

Wiener models ??
difference between black box, white box and gray box modeling of pedals
modeling 3 distortion effects using black box modeling with wavenet
their linear impulse responses were estimated using the **swept-sine technique**
Wavenet was configured to be really fast so it can be used real-time

trained by minimizing the error-to-signal ratio (ESR)
this means that some frequencies with lower energies contribute the same to the loss as other higher energy frequencies
Impulse response ???


### RNNs Experiment

Faster than Wavenet, simmilar in accuracy
In RTBBMWRnns it also conditioned tone knobs, but in this work they did not

## GASCAR 2022
(Genetically automated synthesizer configuration for audio replication)
(Bachelor thesis)

synthesizer types:
- subtractive
- additive
- frequency modulation
- wavetable

(DEAP)[https://deap.readthedocs.io/en/master/]
Is 2011 paper cited in this?

## Sound Resynthesis with a Genetic Algorithm 2011
(Final year project)

Masking genes (conditioning other genes to active/inactive), I don't think this is necessary
Normalizing audio before fitness (good idea)

## UNET

skip connections != residual cons

## DDSP

ovo je retardirano: https://github.com/google/gin-config
bolestan rad

- spectral modeling synthesis https://en.wikipedia.org/wiki/Spectral_modeling_synthesis
Feature-Based Synthesis: Mapping Acoustic and Perceptual
Features onto Synthesis Parameters: https://soundlab.cs.princeton.edu/publications/2006_icmc_fbs.pdf

huang14knobs

https://arxiv.org/abs/1704.03809

https://google-research.github.io/seanet/musiclm/examples/


## DIFFMOOG: A DIFFERENTIABLE MODULAR SYNTHESIZER FOR SOUND MATCHING

introduces a real modular synthesizer for sound matching
more complex chains failed systematically (FM modulations)
novel approach, great possibilities, but fails as for now

## NABLAFX: A FRAMEWORK FOR DIFFERENTIABLE BLACK-BOX AND GRAY-BOX MODELING OF AUDIO EFFECTS

introduces some differentiable audio effects and a great framework for reusing them
Phase Inversion
Gain 
DC Offset
Lowpass/Highpass (second order)
Low/High Shelf (second order)
Peak/Notch
Parametric EQ 
Shelving EQ
Static FIR Filter
Tanh Nonlinearity 
Static MLP Nonlinearity 
Static Rational Nonlinearity

what is FiLM?

https://stackoverflow.com/questions/61132574/can-i-convert-spectrograms-generated-with-librosa-back-to-audio

## REAL-TIME BLACK-BOX MODELLING WITH RECURRENT NEURAL NETWORKS

## CREPE

32 epochs
500  batches
32 examples (batch size)
1024 input size (raw audio samples)

dataset sample size: 16384000 = 371 seconds


## Diffmoog

trains on nsynth from google which is a dataset of labeled pitches and audio
fails to converge on complex lfo based synthesis

nsynth was a predecesor to ddsp which tries to convay timbre of an instrument via wavenet autoencoder (ddsp was based on differentiable synthisizer to aproximate)


## MusicLM

That is, we use the MuLan embeddings computed from the
audio as conditioning during training, while we use MuLan
embeddings computed from the text input during inference.

based on audioLM

MuLan embeddings → MusicLM → discrete tokens → SoundStream decode → mixed waveform

TODO: SoundStream ??



## AUTOMATIC MULTITRACK MIXING WITH A DIFFERENTIABLE MIXING CONSOLE OF NEURAL AUDIO EFFECTS

uses a proxy to make parametrized eq,compresion,gain effects by training WaveNet-like model

this way controls are differentiable and can be tweaked later with frozen weights

controller network, encoder decoder with context, after that they feed to transformation network
works on spectrogram-based VGGish model

results:
very bad subjective results.

## Style Transfer of Audio Effects with Differentiable Signal Processing

has a smart data generation method to create pairs for style transfers

original tracks are fed throught randomly initialized dsp blocks in pairs (one for reference style track and one for target track)

these are then fed as STFT in 2 encoder blocks that change parameter of differentiable audio effects
they try different method for making differentiable audio effects
1 Neural Proxy Pre-training (training TCN with FiLM conditioning)
2 differentiable signal processing (eq and compressor)

results in a decent style transfer (at least based on the numbers alone)

## AUTOMATIC DJ TRANSITIONS WITH DIFFERENTIABLE AUDIO EFFECTS AND GENERATIVE ADVERSARIAL NETWORKS

differentiable eq and gain for tracks, cue points are chosen beforehand for easier mixing, tempo is also detected beforehand

works on generator that first converts audio to spectrograms, then three residual convolutional blocks for encoder with some post processing that are fed as parameters to differentiable track mixing blocks

learns on a lot of dj transition data

performs terrible, not better or worse than rule based and linear models, but significantly worse than human based



## GLU
gated linear unit
GLU(X) = (X * W + b) elementwise_multiplied_by sigmoid(X * V + c)


## WAVE-U-NET: A MULTI-SCALE NEURAL NETWORK FOR END-TO-END AUDIO SOURCE SEPARATION 2018

uses wavenet style model but with skip connections to retain phase information
is trained on audio waveform alone

## SPLEETER 2019

uses similar U net network for singing voice separation, but greater dataset, has state of the art performance in 2019

## Music Source Separation in the Waveform Domain 2021 (Demucs)

uses wavenet like model

6 decoder and encoder layers with linear and lstm layers in the middle
uses GLU
major contribution to wave u net because it changed encoder decoder structure
simmilar in results to D3Net

## Hybrid Spectrogram and Waveform Source Separation 2021

is basically DEMUCS but trained on both spectrograms and pure audio waveform, with different encoders and decoders (6 of them) that concat results in the middle

The original Hybrid Demucs model
is made of two U-Nets, one in the time domain (with temporal convolutions) and one in the spectrogram domain (with
convolutions over the frequency axis). Each U-Net is made
of 5 encoder layers, and 5 decoder layers. After the 5-th encoder layer, both representation have the same shape, and they
are summed before going into a shared 6-th layer.


## HYBRID TRANSFORMERS FOR MUSIC SOURCE SEPARATION 2022

uses architecture based on Hybrid Demucs, which 

Hybrid Transformer Demucs keeps the outermost 4 layers as is from the original architecture, and replaces the 2 innermost layers in the encoder and the decoder, including local attention and bi-LSTM, with a cross-domain Transformer
Encode

outperforms Hybrid Demucs by 0.45 dB in SDR

## LALAI

uses perseus which is based on transformer architecture, likely Hybrid Demucs

## Stem n jam

transformer + mamba layers, possibly simmilar to hybrid transfomrer demux





