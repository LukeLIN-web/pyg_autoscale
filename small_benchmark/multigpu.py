import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.distributed as dist
import copy
import os
from torch_geometric.loader import NeighborLoader
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric_autoscale import (get_data, metis, permute, models,
                                       SubgraphLoader, compute_micro_f1)
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel
torch.manual_seed(123)
criterion = torch.nn.CrossEntropyLoss()


def train(run, model, loader, optimizer, grad_norm=None):
    model.train()

    total_loss = total_examples = 0
    for batch, batch_size, n_id, _, _ in loader:
        batch = batch.to(model.device)
        n_id = n_id.to(model.device)

        mask = batch.train_mask[:batch_size]
        mask = mask[:, run] if mask.dim() == 2 else mask
        if mask.sum() == 0:
            continue

        optimizer.zero_grad()
        out = model(batch.x, batch.adj_t, batch_size, n_id)
        loss = criterion(out[mask], batch.y[:batch_size][mask])
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        total_loss += float(loss) * int(mask.sum())
        total_examples += int(mask.sum())

    return total_loss / total_examples


@torch.no_grad()
def test(run, model, data):
    model.eval()

    val_mask = data.val_mask
    val_mask = val_mask[:, run] if val_mask.dim() == 2 else val_mask

    test_mask = data.test_mask
    test_mask = test_mask[:, run] if test_mask.dim() == 2 else test_mask

    out = model(data.x, data.adj_t)
    val_acc = compute_micro_f1(out, data.y, val_mask)
    test_acc = compute_micro_f1(out, data.y, test_mask)

    return val_acc, test_acc


def run(rank, world_size, conf):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    device = torch.device(rank)
    model_name, dataset_name = conf.model.name, conf.dataset.name
    conf.model.params = conf.model.params[dataset_name]
    params = conf.model.params
    if rank == 0:
        print(OmegaConf.to_yaml(conf))
    if isinstance(params.grad_norm, str):
        params.grad_norm = None

    data, in_channels, out_channels = get_data(conf.root, dataset_name)
    # Split training indices into `world_size` many chunks:
    kwargs = dict(batch_size=1024, num_workers=4, persistent_workers=True)
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    train_loader = NeighborLoader(data, input_nodes=train_idx,
                                  num_neighbors=[25, 10], shuffle=True,
                                  drop_last=True, **kwargs)

    data = data.clone().to(device)  # Let's just store all data on GPU...

    GNN = getattr(models, model_name)
    model = GNN(
        num_nodes=data.num_nodes,  # 这里要不要分割?
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,  # ... and put histories on GPU as well.
        **params.architecture,
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])

    results = torch.empty(params.runs)
    if rank == 0:
        pbar = tqdm(total=params.runs * params.epochs)
    for run in range(params.runs):
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

        test(0, model, data)  # Fill history.

        best_val_acc = 0
        for epoch in range(params.epochs):
            train(run, model, train_loader, optimizer, params.grad_norm)
            val_acc, test_acc = test(run, model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                results[run] = test_acc
            if rank == 0:
                pbar.set_description(f'Mini Acc: {100 * results[run]:.2f}')
                pbar.update(1)
    if rank == 0:
        pbar.close()
    print(f'Mini Acc: {100 * results.mean():.2f} ± {100 * results.std():.2f}')


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(conf):
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, conf), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
