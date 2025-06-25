import numpy as np
import random
from pedalboard import Pedalboard, Chorus, Distortion, Reverb, Delay, Compressor
import inspect
from pedalboard.io import AudioFile

MAX_POPULATION = 50 #Max population size
TOURNAMENT_SIZE = 4 #Tournament size for selection method
ELITISM_RATE = 0.01 #Elitism rate for selection method



def tournament_selection(population, tournament_size=3, selection_probability=0.8):
    tournament = random.sample(population, tournament_size)
    tournament.sort()
    
    if random.random() < selection_probability:
        return tournament[0]
    else:
        return random.choice(tournament)
    
import numpy as np
import random
from typing import List, Type
from pedalboard import Pedalboard, Chorus, Distortion, Reverb, Delay, Compressor
from pedalboard.io import AudioFile

# Define all effect classes and their tunable parameters
EFFECTS_CONFIG = {
    Compressor: {
        'threshold_db': (-60, 0),
        'ratio': (1, 20),
        'attack_ms': (0.1, 100),
        'release_ms': (1, 500),
    },
    Distortion: {
        'drive_db': (0, 30),
    },
    Chorus: {
        'rate_hz': (0.1, 5),
        'depth': (0, 1),
        'centre_delay_ms': (1, 30),
        'feedback': (0, 1),
    },
    Reverb: {
        'room_size': (0, 1),
        'damping': (0, 1),
        'wet_level': (0, 1),
        'dry_level': (0, 1),
        'width': (0, 1),
        'freeze_mode': (0, 1),
    },
    Delay: {
        'delay_seconds': (0, 2),
        'feedback': (0, 1),
    }
}

def random_param(low, high):
    return np.random.uniform(low, high)

def loss(original, effected):
    return np.mean((original - effected) ** 2)

class EffectWrapper:
    def __init__(self, effect_cls: Type, params: dict = None, dry_wet_mix: float = None):
        self.effect_cls = effect_cls
        self.param_ranges = EFFECTS_CONFIG[effect_cls]
        self.params = {
            key: random_param(*bounds)
            for key, bounds in self.param_ranges.items()
        } if params is None else params
        self.dry_wet_mix = random_param(0, 1) if dry_wet_mix is None else dry_wet_mix
        self.effect_instance = self._instantiate()

    def _instantiate(self):
        try:
            return self.effect_cls(**self.params)
        except Exception as e:
            raise ValueError(f"Invalid parameters for {self.effect_cls.__name__}: {e}")

    def update_instance(self):
        self.effect_instance = self._instantiate()

    def apply(self, audio_chunk, sample_rate):
        # Apply plugin, then apply dry/wet mix externally
        processed = self.effect_instance(audio_chunk, sample_rate)
        return mix_dry_wet(audio_chunk, processed, self.dry_wet_mix)

    def mutate(self, mutation_strength=0.1):
        # Mutate a random parameter or dry/wet mix
        if random.random() < 0.5 and self.params:
            param_to_mutate = random.choice(list(self.params.keys()))
            low, high = self.param_ranges[param_to_mutate]
            original_value = self.params[param_to_mutate]

            # Scale or offset mutation
            if random.random() < 0.5:
                mutated_value = original_value * (1 + (0.5 - random.random()) * mutation_strength)
            else:
                mutated_value = original_value + (0.5 - random.random()) * (high - low) * mutation_strength

            self.params[param_to_mutate] = np.clip(mutated_value, low, high)
        else:
            # Mutate dry_wet_mix
            mutated_value = self.dry_wet_mix + (0.5 - random.random()) * mutation_strength
            self.dry_wet_mix = float(np.clip(mutated_value, 0, 1))

        self.update_instance()

    def copy(self):
        return EffectWrapper(self.effect_cls, self.params, self.dry_wet_mix)



class Obj:
    def __init__(self, rand=False):
        self.effects: List[EffectWrapper] = []
        self.loss = float('inf')
        if rand:
            for effect_cls in EFFECTS_CONFIG:
                self.effects.append(EffectWrapper(effect_cls))

    def calc_loss(self, input_audio: str, target_audio: str) -> 'Obj':
        self.loss = 0

        with AudioFile(input_audio) as dry_file, AudioFile(target_audio) as wet_file:
            sample_rate = dry_file.samplerate

            while dry_file.tell() < dry_file.frames:
                dry_chunk = dry_file.read(sample_rate)
                wet_chunk = wet_file.read(sample_rate)

                if dry_chunk.shape != wet_chunk.shape:
                    raise ValueError("Input and target chunks must have the same shape")

                processed = dry_chunk.copy()
                for effect in self.effects:
                    processed = effect.apply(processed, sample_rate)

                self.loss += loss(wet_chunk, processed)

        return self



    def mate(self, other: 'Obj') -> 'Obj':
        child = Obj(rand=False)
        child.effects = [
            random.choice([self.effects[i], other.effects[i]]).copy()
            for i in range(len(self.effects))
        ]
        return child

    def mutate(self, mutation_chance=0.3):
        for effect in self.effects:
            if random.random() < mutation_chance:
                effect.mutate()
        return self

    def __lt__(self, other: 'Obj'):
        return self.loss < other.loss


def mix_dry_wet(dry, wet, mix=0.5):
    assert 0 <= mix <= 1
    return (1 - mix) * dry + mix * wet




def main():
    board = Pedalboard([Distortion(drive_db=50), Chorus(), Reverb(room_size=0.25)])
    with AudioFile('input.wav') as f:
        with AudioFile('output.wav', 'w', f.samplerate, f.num_channels) as o:
    
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                
                # Run the audio through our pedalboard:
                effected = board(chunk, f.samplerate, reset=False)
                
                # Write the output to our output file:
                o.write(effected)

    
    population = [Obj(rand=True).calc_loss("input.wav", "output.wav") for _ in range(MAX_POPULATION)]

    it = 0
    while True:
        population.sort()
        new_population = population[:int(MAX_POPULATION*ELITISM_RATE)]
        while len(new_population) < MAX_POPULATION:
            parent1 = tournament_selection(population, tournament_size=TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, tournament_size=TOURNAMENT_SIZE)

            child = parent1.mate(parent2)
            child.mutate()
            child.calc_loss("input.wav", "output.wav")

            new_population.append(child)

        population = new_population
        
        total_loss= sum([i.loss for i in population])
        print(f"{it}: {total_loss/MAX_POPULATION}") # average loss accross population

        best = population[0]
        with AudioFile('input.wav') as f:
            with AudioFile(f'gen/output_gen_{it}.wav', 'w', f.samplerate, f.num_channels) as o:

                while f.tell() < f.frames:
                    chunk = f.read(f.samplerate)
                    processed = chunk.copy()
                    for effect in best.effects:
                        processed = effect.apply(processed, f.samplerate)
                    o.write(processed)


        it += 1

if __name__ == "__main__":
    main()
