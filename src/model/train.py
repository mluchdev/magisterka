import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from common import auroc, tpr_at_fpr
from data.observer_data import TokenDataset
from model.mamba_detector import MambaDetector

CFG = dict(split=sys.argv[1] if len(sys.argv) > 1 else "test", d=64, n_layers=2, max_len=1024, batch_size=16, lr=1e-3, epochs=5, seed=0)


@torch.no_grad()
def metrics(model, loader, groups):
    """Class accuracy / recall, plus human vs machine from the same softmax: score = log P(machine) - log P(human)."""
    model.eval()
    out = [(model(x.cuda(), cu.cuda()).float().cpu(), y, g) for x, cu, y, g in loader]
    l, y, g = (torch.cat(t) for t in zip(*out))
    logp, y, g, h = torch.log_softmax(l, 1).numpy(), y.int().numpy(), g.numpy(), groups.index("human")
    pred = logp.argmax(1)
    s = np.logaddexp.reduce(np.delete(logp, h, 1), 1) - logp[:, h]
    rec = {f"recall/{n}": (pred[g == c] == c).mean() for c, n in enumerate(groups)}
    return dict(acc=(pred == g).mean(), balanced_acc=np.mean(list(rec.values())), binary_acc=((pred != h) == y).mean(),
                auroc=auroc(y, s), tpr_at_1fpr=tpr_at_fpr(y, s), **rec)


def main():
    torch.manual_seed(CFG["seed"])
    data = {p: TokenDataset(CFG["split"], p, CFG["max_len"]).loader(CFG["batch_size"]) for p in ("train", "val", "test")}
    groups = data["train"].dataset.groups
    model = MambaDetector(len(groups), CFG["d"], CFG["n_layers"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    wandb.init(project="llm-origin-observer", name="mamba-detector", config=CFG | {"classes": groups})
    step = 0
    for ep in range(CFG["epochs"]):
        model.train()
        for x, cu, y, g in data["train"]:
            loss = F.cross_entropy(model(x.cuda(), cu.cuda()), g.cuda())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
            if step % 20 == 0:
                wandb.log({"train/loss": loss.item()}, step=step)
        for part in ["val"] + ["test"] * (ep == CFG["epochs"] - 1):
            m = metrics(model, data[part], groups)
            wandb.log({f"{part}/{k}": v for k, v in m.items()}, step=step)
            print(part, ep + 1, {k: round(float(v), 4) for k, v in m.items()}, flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
