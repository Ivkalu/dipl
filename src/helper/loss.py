import torch

# Pre-emphasis filter funkcija
def pre_emphasis_filter(x, coeff=0.95):
    # x je tensor shape (batch_size, seq_len)
    # primeni y[t] = x[t] - coeff * x[t-1]
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1:] = x[:, 1:] - coeff * x[:, :-1]
    return y

# Custom loss: Error to signal ratio sa pre-emphasis filterom
def error_to_signal(y_true, y_pred):
    #y_true = pre_emphasis_filter(y_true)
    #y_true = pre_emphasis_filter(y_pred)
    numerator = torch.sum((y_true - y_pred)**2, dim=1)
    denominator = torch.sum(y_true**2, dim=1) + 1e-10
    loss = numerator / denominator
    return torch.mean(loss)


def mean_squared_error(y_true, y_pred):
    loss = torch.mean((y_true - y_pred) ** 2)
    return loss

def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))