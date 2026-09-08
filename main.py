import argparse
import math
import time
import net
import pandas as pd
import torch
import torch.nn as nn
from net import magnn
import numpy as np
from torch.serialization import safe_globals
from util import *
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import random
import numpy as np
import torch

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    for X_batch, Y_batch in data.get_batches(X, Y, batch_size, True):
        batch_size_curr = X_batch.shape[0]
        X_batch = X_batch.view(batch_size_curr, args.in_dim, args.num_nodes, args.seq_in_len)
        model.zero_grad()
        output, _ = model(X_batch)
        scale_expanded = data.scale.view(1, 1, -1, 1).expand(
            batch_size_curr, args.seq_out_len, args.num_nodes, 1
        )
        loss = criterion(output * scale_expanded, Y_batch.unsqueeze(-1) * scale_expanded)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        n_samples += batch_size_curr * args.num_nodes * args.seq_out_len

        if iter % 100 == 0:
            print('iter:{:3d} | loss: {:.6f}'.format(
                iter, loss.item() / (batch_size_curr * args.num_nodes * args.seq_out_len)))
        iter += 1

    return total_loss / n_samples


def evaluate(data, X, Y, model, batch_size):
    model.eval()
    total_loss = 0.0
    total_loss_l1 = 0.0
    n_samples = 0

    for X_batch, Y_batch in data.get_batches(X, Y, batch_size, False):
        batch_size_curr = X_batch.shape[0]
        X_batch = X_batch.view(batch_size_curr, args.in_dim, args.num_nodes, args.seq_in_len)
        with torch.no_grad():
            output, _ = model(X_batch)
        output = output.squeeze(-1)
        scale = data.scale.view(1, 1, -1).expand(batch_size_curr, args.seq_out_len, args.num_nodes)
        y_pred_scaled = output * scale
        y_true_scaled = Y_batch * scale
        total_loss += ((y_pred_scaled - y_true_scaled) ** 2).sum().item()
        total_loss_l1 += (torch.abs(y_pred_scaled - y_true_scaled)).sum().item()
        n_samples += batch_size_curr * args.num_nodes * args.seq_out_len

    mse = total_loss / n_samples
    mae = total_loss_l1 / n_samples
    return mae, mse

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--subgraph_size', type=int, default=8, help='k')
parser.add_argument('--gcn_depth', type=int, default=5, help='graph convolution depth')
parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--data', type=str, default='databohai.csv', help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=False)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--num_nodes', type=int, default=300,help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--conv_channels', type=int, default=64, help='convolution channels')
parser.add_argument('--scale_channels', type=int, default=16, help='scale channels')
parser.add_argument('--end_channels', type=int, default=16, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=30, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=15, help='output sequence length')
parser.add_argument('--horizon', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')
parser.add_argument('--epochs', type=int, default=500, help='')
parser.add_argument('--early_stop_patience', type=int, default=30, help='early stopping patience')
parser.add_argument('--skip_sliding', action='store_true', help='skip sliding window evaluation')
parser.add_argument('--sliding_metrics_file', type=str, default=None,
                    help='path to save sliding window daily metrics csv')
parser.add_argument('--sliding_start_day', type=int, default=1881, help='sliding window start day')
parser.add_argument('--no_sliding_pred_save', action='store_true',
                    help='skip saving sliding window predictions and ground truth')
parser.add_argument('--data_dir', type=str, default=None, help='directory containing the data csv')
parser.add_argument('--result_file', type=str, default=None, help='save final metrics as json')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args, unknown = parser.parse_known_args()
device = torch.device(args.device)
torch.set_num_threads(3)
class Optim(object):
    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)
    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._makeOptimizer()
    def step(self):
        grad_norm = 0
        if self.clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()
        return grad_norm

    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True
        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            self._makeOptimizer()
        self.start_decay = False
        self.last_ppl = ppl

def evaluate_sliding_windows(
    model, data, device,
    start_day=1881, end_day=None,
    save_path="sliding_metrics.csv",
    pred_save_path="all_predictions.csv",
    true_save_path="all_ground_truth.csv",
    save_predictions=True,
):
    import numpy as np
    import torch
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import pandas as pd
    seq_len = 30
    pred_len = 15
    raw_data = data.dat
    original_data = data.raw_data if hasattr(data, "raw_data") else data.dat
    if end_day is None:
        end_day = len(raw_data) - pred_len
    scale = data.scale.cpu().numpy() if torch.is_tensor(data.scale) else data.scale  # (num_nodes,)

    daily_mse = [[] for _ in range(pred_len)]
    daily_mae = [[] for _ in range(pred_len)]

    all_preds = [] 
    all_truth = []  

    count = 0

    for cur_start in range(start_day, end_day + 1):  
        x_seq = raw_data[cur_start - seq_len:cur_start]          
        y_true = original_data[cur_start:cur_start + pred_len] 

        if x_seq.shape[0] != seq_len:
            continue

        x_input = torch.tensor(x_seq, dtype=torch.float).to(device)
        x_input = x_input.permute(1, 0).unsqueeze(0).unsqueeze(0)  
        model.eval()
        with torch.no_grad():
            output = model(x_input)
            y_pred = output[0] if isinstance(output, tuple) else output

        y_pred = y_pred[0].squeeze(-1).cpu().numpy() 

        y_pred_inv = y_pred * scale[np.newaxis, :] 
        y_true_inv = y_true 
    

        for day in range(pred_len):
            mse = mean_squared_error(y_true_inv[day], y_pred_inv[day])
            mae = mean_absolute_error(y_true_inv[day], y_pred_inv[day])
            daily_mse[day].append(mse)
            daily_mae[day].append(mae)

        for day in range(pred_len):
            for node in range(y_pred_inv.shape[1]):
                all_preds.append([count, day + 1, node, y_pred_inv[day, node]])
                all_truth.append([count, day + 1, node, y_true_inv[day, node]])

        count += 1

    print(f"\nCompleted {count} sliding evaluations from day {start_day} to {end_day}.\n")

    avg_mse = [np.mean(m) for m in daily_mse]
    avg_mae = [np.mean(m) for m in daily_mae]

    overall_mse = np.mean(avg_mse)
    overall_mae = np.mean(avg_mae)

    print("Average Per-Day Errors (Over All Sliding Windows)")
    for i in range(pred_len):
        print(f"Day {i+1:02d} - MSE: {avg_mse[i]:.4f}, MAE: {avg_mae[i]:.4f}")

    print(f"Overall 15-Day Average MSE: {overall_mse:.9f}")
    print(f"Overall 15-Day Average MAE: {overall_mae:.9f}")


    df_metrics = pd.DataFrame({
        "Day": list(range(1, pred_len + 1)),
        "Avg_MSE": avg_mse,
        "Avg_MAE": avg_mae
    })

    df_metrics.loc[len(df_metrics)] = ["Overall(15d)", overall_mse, overall_mae]

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df_metrics.to_csv(save_path, index=False)
        print(f"Saved sliding window metrics to: {save_path}")

    if save_predictions:
        df_preds = pd.DataFrame(all_preds, columns=["Window", "Day", "Node", "Pred_Value"])
        df_truth = pd.DataFrame(all_truth, columns=["Window", "Day", "Node", "True_Value"])
        if pred_save_path:
            os.makedirs(os.path.dirname(pred_save_path) or ".", exist_ok=True)
            df_preds.to_csv(pred_save_path, index=False)
            print(f"Saved predictions to: {pred_save_path}")
        if true_save_path:
            os.makedirs(os.path.dirname(true_save_path) or ".", exist_ok=True)
            df_truth.to_csv(true_save_path, index=False)
            print(f"Saved ground truth to: {true_save_path}")

    return {
        "daily_mse": [float(v) for v in avg_mse],
        "daily_mae": [float(v) for v in avg_mae],
        "overall_mse": float(overall_mse),
        "overall_mae": float(overall_mae),
        "num_windows": count,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_path(data_file, data_dir=None):
    if data_dir:
        return os.path.join(data_dir, data_file)
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    local_path = os.path.join(base_dir, data_file)
    if os.path.isfile(local_path):
        return local_path
    cwd_path = os.path.join(os.getcwd(), data_file)
    if os.path.isfile(cwd_path):
        return cwd_path
    return os.path.join("/kaggle/input/datasets/yuanyujing/magnndata6", data_file)


def main(params):
    set_seed(args.seed)
    best_val = float('inf')
    best_epoch = 0
    patience = args.early_stop_patience
    train_start = time.time()

    dropout = params.get('dropout', 0.1)
    subgraph_size = params.get('subgraph_size', args.subgraph_size)
    conv_channels = params.get('conv_channels', 64)
    scale_channels = conv_channels
    gnn_channels = conv_channels
    data_path = resolve_data_path(args.data, args.data_dir)

    Data = DataLoaderS(data_path, 0.6, 0.2, device, args.horizon, window=30, normalize=args.normalize)

    model = magnn(
        args.gcn_depth, args.num_nodes,
        device,
        node_dim=args.node_dim,
        subgraph_size=subgraph_size,
        dropout=dropout,
        conv_channels=conv_channels,
        scale_channels=scale_channels,
        end_channels=args.end_channels,
        gnn_channels=gnn_channels,
        seq_length=30,
        in_dim=args.in_dim,
        out_dim=args.seq_out_len,
        propalpha=args.propalpha,
        tanhalpha=args.tanhalpha,
        single_step=False,
        num_scales=4
    ).to(device)

    nParams = sum([p.nelement() for p in model.parameters()])
    print(f'Model initialized with {nParams:,} parameters', flush=True)

    criterion = nn.L1Loss(reduction='sum').to(device) if args.L1Loss else nn.MSELoss(reduction='sum').to(device)
    optim = Optim(model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay)

    try:
        print('\nBegin Training...\n')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            val_mae, val_mse = evaluate(Data, Data.valid[0], Data.valid[1], model, args.batch_size)

            print(f"| Epoch {epoch:3d} | time: {time.time()-epoch_start_time:5.2f}s "
                  f"| train_loss {train_loss:5.4f} | val_mae {val_mae:5.4f} | val_mse {val_mse:5.4f}")

            if val_mse < best_val:
                best_val = val_mse
                best_epoch = epoch
                os.makedirs(os.path.dirname(args.save), exist_ok=True)
                torch.save(model.state_dict(), args.save)
                print(f"New best model saved (epoch {epoch}, val_mse={val_mse:.4f})", flush=True)
            else:
                if epoch - best_epoch >= patience:
                    print(f"Early stopping at epoch {epoch}, no improvement for {patience} epochs.")
                    break

            if epoch % 5 == 0:
                test_mae, test_mse = evaluate(Data, Data.test[0], Data.test[1], model, args.batch_size)
                print(f"Test mae {test_mae:5.4f} | test mse {test_mse:5.4f}", flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Training interrupted manually.')

    print(f"\nLoading best model from epoch {best_epoch} for final evaluation...")
    model.load_state_dict(torch.load(args.save))
    model = model.to(device)

    v_mae, v_mse = evaluate(Data, Data.valid[0], Data.valid[1], model, args.batch_size)
    t_mae, t_mse = evaluate(Data, Data.test[0], Data.test[1], model, args.batch_size)
    print(f"\nFinal Valid MAE: {v_mae:.4f} | MSE: {v_mse:.4f}")
    print(f"Final Test  MAE: {t_mae:.4f} | MSE: {t_mse:.4f}")

    sliding_metrics = None
    if not args.skip_sliding:
        sliding_save = args.sliding_metrics_file or "metrics.csv"
        pred_save = None if args.no_sliding_pred_save else "predictions.csv"
        truth_save = None if args.no_sliding_pred_save else "ground_truth.csv"
        sliding_metrics = evaluate_sliding_windows(
            model, Data, device,
            start_day=args.sliding_start_day,
            save_path=sliding_save,
            pred_save_path=pred_save,
            true_save_path=truth_save,
            save_predictions=not args.no_sliding_pred_save,
        )

    results = {
        "subgraph_size": subgraph_size,
        "gcn_depth": args.gcn_depth,
        "propalpha": args.propalpha,
        "best_epoch": best_epoch,
        "val_mae": float(v_mae),
        "val_mse": float(v_mse),
        "test_mae": float(t_mae),
        "test_mse": float(t_mse),
        "train_time_sec": round(time.time() - train_start, 2),
    }
    if sliding_metrics:
        results["sliding_overall_mse"] = sliding_metrics["overall_mse"]
        results["sliding_overall_mae"] = sliding_metrics["overall_mae"]
        results["sliding_num_windows"] = sliding_metrics["num_windows"]
        for i in range(len(sliding_metrics["daily_mse"])):
            d = i + 1
            results[f"sliding_day{d}_mse"] = sliding_metrics["daily_mse"][i]
            results[f"sliding_day{d}_mae"] = sliding_metrics["daily_mae"][i]

    if args.result_file:
        import json
        os.makedirs(os.path.dirname(args.result_file) or ".", exist_ok=True)
        with open(args.result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.result_file}")

    print("\nBest model used for final evaluation.")
    return results

if __name__ == "__main__":
    main({})
