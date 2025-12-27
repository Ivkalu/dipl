# Audio research

## GuitarML

This is used for notes about GuitarML guy that made simulator for amps and pedals using ML.

- [youtube channel](https://www.youtube.com/watch?v=xkrqF0D8pfQ)
- [website](https://guitarml.com/)
- [github](https://github.com/mlamsk)
- [blog 1](https://towardsdatascience.com/transfer-learning-for-guitar-effects-4af50609dce1)
- [blog 2](https://medium.com/nerd-for-tech/neural-networks-for-real-time-audio-introduction-ed5d575dc341)

He used multiple approaches:
- [LSTM](https://github.com/GuitarML/GuitarLSTM)
- [WaveNet](https://github.com/GuitarML/SmartGuitarAmp)

He also exported model to RaspberryPi:
- [NeuralPi](https://github.com/GuitarML/NeuralPi)

He exported all models to VST using JUCE.

As noted in [SmartGuitarPedal](https://github.com/GuitarML/SmartGuitarPedal), the WaveNet model is effective at emulating distortion style effects or tube amplifiers, but cannot capture time based effects such as reverb or delay.

The problem seems to be that time based effects require some sort of memory, or context, maybe RNNs or GPTs would work better on this type of task. 

# Genetic algorithm in recreating audio effects

But these models need to work their way up to recreating complex reverb and distortion effects. What if in combination with this, they need to also model a distortion on top of that, or any more complex effect. Why don't we help them somehow. We have to rethink the task.

Can we reuse existing reverb and other pedals to recreate more complex models using AI? I suggest following architecture:

```
Input => Wavenet(Input) => Output
```

```
Input => a*effectA(Input) + b*effectB(Input) + c*effectC(Input) + ... => Output
```

We can have an arbitrary number of these effects so model can combine them in a way to create a new effect. Which effects we can recreate using this method varies on expressivness of "primal effects" used in model. 
Problem with this approach is that we lose the gradient, so we have to change the training approach.
- How about a genetic algorithm?
- Do we lose gradient actually? Can we create a subset of effects that are differentiable. Distortion, compression and simmilar effects can be easily differentiable since they are usually just a combination of multiplication and max functions. Delays and reverb need context.

Well we have this formula that can easily be translated to an audio synthesis.


## Genetic algorithm in audio synthesis

```
MIDI => effects(a*OscA(MIDI) + b*OscB(MIDI) + c*OscC(MIDI)) => Output
```

So we can use a simmilar model to recreate audio synthisiser. Another upgrade would be to make it unsupervised. So we only have output file. This would be great so model can learn to recreate synth patches from other authors. This would also need some kind of loss function to figure out the pitch difference, not only the sound difference.

One experemint could involve just the learning of MIDI notes, without complex effects and see if genetic algorithm is even capable of doing that.

Also another question: Could we throw in a NEAT just for fun? Okay, not just for fun, but we could then recreate any modular synth patch. Problem would probably be much greater complexity and time to learn.

### Synthetizing drum using drum synthisizers

Same as for regular audio synthesis, but different synthesizer will be used with different params.


## Other Research

- How does shazam work
- How does google recognize your melodies you hum to him
- Which machine learning technologies does spotify use
- Which models are used to split audio SOTA
- Which models are used to remove noise from audio SOTA
- Which models are used to generate speech  SOTA
- Which models are used to translate speech SOTA
- Automatic mixing with AI - ozone
- How do algorithms for finding notes from audio work
- Diffusion models
- DeepSeek?
- How is Google's walking dog robot connected to AI and GA?