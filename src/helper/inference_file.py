import torch


def process_whole_audio_file(model, x, seq2seq, device='cpu'):
    x = x.to(device)
    model.eval()
    if seq2seq:
        y = model(x.permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
    else:
        model.seq2seq = True
        y = model(x.permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        model.seq2seq = seq2seq
    return y
    
    