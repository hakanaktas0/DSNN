import torch


def complex_merged_dropout(x,p=0.9,training=True):
    if x.is_complex():
        mask = torch.nn.functional.dropout(torch.ones_like(x.real),p=p,training=training)
        return x * mask
    else:
        return torch.nn.functional.dropout(x,p=p,training=training)

def complex_separate_dropout(x,p=0.9,training=True):
    real = torch.nn.functional.dropout(x.real,p=p,training=training)
    imag = torch.nn.functional.dropout(x.imag,p=p,training=training)
    return torch.complex(real,imag)