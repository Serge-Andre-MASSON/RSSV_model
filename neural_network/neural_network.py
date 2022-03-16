from torch import nn
from torch.optim.adam import Adam


class InverseProblem(nn.Module):
    def __init__(self, number_of_calls, number_of_parameters):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(number_of_calls, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, number_of_parameters)
        )

    def forward(self, xb):
        return self.lin(xb.float())


def get_optimized_model(number_of_calls, number_of_parameters, learning_rate=0.05):
    model = InverseProblem(number_of_calls, number_of_parameters)
    opt = Adam(model.parameters(), lr=learning_rate)

    return model, opt


def loss_batch(model: InverseProblem, loss_func, xb, yb, opt: Adam = None):
    loss = loss_func(model(xb.float()), yb).float()

    if opt:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(model: InverseProblem, train_dl, epochs, opt=Adam, loss_func=nn.MSELoss()):

    for _ in range(epochs):

        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb.float(), yb.float(), opt)
