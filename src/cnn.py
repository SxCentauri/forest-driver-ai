import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_ch=4, n_cls=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16->8
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 8->4
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def train_cnn(Xtr, ytr, Xte, epochs=15, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        total = 0.0
        for i in range(0, len(yt), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward(); opt.step()
            total += loss.item() * len(b)
        print(f"  epoch {ep + 1:02d}/{epochs}  loss={total / len(yt):.4f}")
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xte)).argmax(1).numpy()
