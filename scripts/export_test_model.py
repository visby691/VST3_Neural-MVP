import json
import torch
import torch.nn as nn
from pathlib import Path


class SimpleDistortionNet(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def export_to_rtneural_json(model: SimpleDistortionNet, output_path: str):
    """PyTorchモデルの重みを RTNeural::json_parser::parseJson 互換のJSON形式で保存する。
    ※ RTNeuralはKeras形式の重み配列 (in_features, out_features) を期待するため転置(.T)する。
    """
    model.eval()
    with torch.no_grad():
        w1 = model.fc1.weight.detach().cpu().numpy().T.tolist()
        b1 = model.fc1.bias.detach().cpu().numpy().tolist()
        w2 = model.fc2.weight.detach().cpu().numpy().T.tolist()
        b2 = model.fc2.bias.detach().cpu().numpy().tolist()

    rtneural_json = {
        "in_shape": [None, 1],
        "layers": [
            {
                "type": "dense",
                "activation": "tanh",
                "shape": [None, model.fc1.out_features],
                "weights": [w1, b1],
            },
            {
                "type": "dense",
                "activation": "",
                "shape": [None, 1],
                "weights": [w2, b2],
            },
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rtneural_json, f, indent=2)


def main():
    torch.manual_seed(42)
    model = SimpleDistortionNet(hidden_size=8)

    # ターゲット関数: 音の変化が分かりやすいソフトクリッピング歪み
    x = torch.linspace(-1.0, 1.0, 2000).unsqueeze(1)
    y_target = 0.7 * torch.tanh(6.0 * x)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()

    for epoch in range(1000):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y_target)
        loss.backward()
        optimizer.step()

    print(f"Final MSE Loss: {loss.item():.6f}")

    # スクリプトの1つ上（リポジトリルート）の models/ フォルダを出力先にする
    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "test_distortion.json"

    export_to_rtneural_json(model, str(out_file))
    print(f"Exported: {out_file}")


if __name__ == "__main__":
    main()