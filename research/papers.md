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