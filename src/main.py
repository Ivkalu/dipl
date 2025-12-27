import os
import argparse
import torch
import torch.optim as optim
from datasets.data_module import DataModule, DATASET_PATH 
from helper.loss import error_to_signal, mean_squared_error
from helper.save_wav import save_wav
from helper.train import train, evaluate
from helper.inference_file import process_whole_audio_file

from models.lstm import LSTM
from models.baselineFCNet import FCNet
from models.mamba import Mamba
from models.tcn import TCN
from models.wavenet import WaveNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):


    effect_choice = ["simpleDist", "ragingDemon", "dragonflyPlateReverb", "dragonflyRoomReverb"]
    model_choice = ["fcn", "lstm", "wavenet", "tcn", "mamba"]

    if args.effect.lower() in effect_choice:
        effect_name = args.effect.lower()
    elif args.effect in ["0","1","2","3"]:
        effect_name = effect_choice[int(args.effect)]
    else:
        print(f"Args effect name wrong: {args.effect}")
        return

    if args.model.lower() in model_choice:
        model_name = args.model.lower()
    elif args.model in ["0","1","2","3","4"]:
        model_name = model_choice[int(args.model)]
    else:
        print(f"Model name wrong: {args.model}")
        return

    epochs = args.epochs
    learning_rate = 0.001
    input_size = args.input_size
    guitar_only = args.guitar_only


    seq2seq=False
    if model_name == "fcn":
        model = FCNet(seq2seq=seq2seq).to(device)                  # works seq2one and seq2seq
    elif  model_name == "lstm":
        model = LSTM(seq2seq=seq2seq).to(device)                   # works seq2one and seq2seq
    elif model_name == "wavenet":
        model = WaveNet().to(device)                            # works seq2seq
    elif model_name == "tcn":
        model = TCN().to(device)                                # works seq2seq
    elif model_name == "mamba":
        model = Mamba().to(device)                              # works seq2one and seq2seq
    else:
        return
    
    data_module = DataModule(
        effect_name=effect_name, 
        input_size=input_size, 
        max_wav_files=args.max_wav_files,
        batch_size=args.batch_size, 
        num_workers=0,
        seq2seq=seq2seq,
        guitar_only=guitar_only)

    train_dataloader = data_module.train_dataloader(max_samples=args.train_samples) 
    test_dataloader = data_module.test_dataloader(max_samples=args.test_samples)
    test_dataset = data_module.get_waveform_dataset("test")
    
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training {effect_name} effect with {model_name}")

    model.train()
    train(
        epochs=epochs, 
        data_loader=train_dataloader, 
        optimizer=optimizer,
        model=model, 
        loss_func=mean_squared_error)
    
    del train_dataloader  # free memory
    torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
        
        in_data, out_data = next(iter(test_dataset))

        guitar = "guitar" if guitar_only else "other"
        if not seq2seq: 
            model_name += "seq2one"
        base_path = os.path.join("results", guitar, effect_name, model_name)
        
        try:
            torch.backends.cudnn.enabled = False
            pred = process_whole_audio_file(model, in_data, device=device, seq2seq=seq2seq)

            # definiramo foldere za rezultate
            
            os.makedirs(base_path, exist_ok=True)

            predicted_path = os.path.join(base_path, "predicted.wav")
            input_path = os.path.join("results", guitar, "input.wav")
            output_path = os.path.join("results", guitar, effect_name, "target.wav")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            save_wav(predicted_path, pred.cpu().numpy())
            save_wav(input_path, in_data.cpu().numpy())
            save_wav(output_path, out_data.cpu().numpy())

        except Exception as e: 
            print(f"An error occurred while processing whole wav files: {e}")
            breakpoint()

        try:
            torch.backends.cudnn.enabled = True
            mse = evaluate(
                data_loader=test_dataloader, 
                model=model, 
                loss_func=mean_squared_error)
            print(f"Total MSE loss on a test dataset: {mse}")

            esr = evaluate(
                data_loader=test_dataloader, 
                model=model, 
                loss_func=error_to_signal)
            print(f"Total ESR loss on a test dataset: {esr}")

            results_path = os.path.join(base_path, "results.txt")
            with open(results_path, "w") as f:
                f.write(f"MSE: {mse}\n")
                f.write(f"ESR: {esr}\n")
            
        except Exception as e: 
            print(f"An error occurred while evaluating model: {e}")
            breakpoint()

            # spremanje modela sa torch.jit
        
        model_path = os.path.join(base_path, "model.pt")
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save(model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Failed to export model: {e}")
            breakpoint()


    del test_dataloader  # free memory
    torch.cuda.empty_cache()

# make a function that will load a model from torch jit and run it on a file
def load_jit_model_and_run(model_path, input_tensor, device="cpu", seq2seq=True):
    """
    Funkcija učitava torch.jit model i pokreće ga na zadanom inputu.
    Ovdje se ne poziva, ali je spremna za korištenje.
    """
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    with torch.no_grad():
        output = process_whole_audio_file(model, input_tensor, device=device, seq2seq=seq2seq)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("effect")
    parser.add_argument("model")
    parser.add_argument("--input_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--train_samples", type=int, default=44100*5)
    parser.add_argument("--test_samples", type=int, default=44100*3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_wav_files", type=int, default=11)
    parser.add_argument(
        "--guitar_only",
        action="store_false",   # if user passes --guitar_only, it will set it to False
        default=True,           # default is True
        help="Use only guitar dataset (default: True, add flag to disable)"
    )

    args = parser.parse_args()
    main(args)