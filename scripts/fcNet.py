import torch

class block(torch.nn.Module):
  def __init__(self, D_in, D_out, bnorm=False):
    self.bnorm = bnorm
    super(block, self).__init__()
    self.linear11 = torch.nn.Linear(D_in, D_out)
    self.linear12 = torch.nn.Linear(D_out,D_out)    
    self.linear21 = torch.nn.Linear(D_in, D_out)
    self.linear22 = torch.nn.Linear(D_out,D_out)

  def forward(self, x):
    h11 = self.linear11(x).clamp(min=0)
    h12 = self.linear12(h11)
    h21 = self.linear21(x).clamp(min=0)
    h22 = self.linear22(h21)
    h = h12 * h22
    return h.clamp(min=0)

net = torch.nn.Sequential(
  block(1,3),
  block(3,6),
  block(6,6),
  torch.nn.Linear(6, 3)
)