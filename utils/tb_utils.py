# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


def _tb_advance_global_step(module):
    if hasattr(module, 'global_step'):
        module.global_step += 1
    return module


def _tb_advance_token_counters(module, tensor, verbose=False):
    token_count = getattr(module, 'tb_token_count', None)
    if token_count is not None:
        T = tensor.size(1)
        if token_count.last != T:
            if token_count.last != 0:
                token_count.total += token_count.last
                token_count.sample_idx += 1
            token_count.last = T

        if verbose:
            print(f'>>> T={T}\tlast_T={token_count.last}\tcumsum_T={token_count.total}')
    return module


def _tb_hist(module, tensor, name, verbose=False):
    hist_kw = dict(bins='auto')

    tb_writer = getattr(module, 'tb_writer', None)
    if tb_writer is not None:
        if module.layer_idx == module.num_layers - 1:
            tensor = tensor[:, 0]

        # per-tensor
        layer_s = str(1 + module.layer_idx).zfill(2)
        full_name = f'{layer_s}/layer/{name}'
        global_step = module.global_step
        if verbose:
            stats = f'min={tensor.min():.1f}, max={tensor.max():.1f}'
            info = (
                f'TB logging {full_name}\t{tuple(tensor.size())}\t({stats})\t'
                f'[global_step={global_step}] ...'
            )
            print(info)
        tb_writer.add_histogram(full_name, tensor, global_step=global_step, **hist_kw)

        # per-token
        sample_idx_s = str(module.tb_token_count.sample_idx + 1).zfill(2)
        T = tensor.size(1)
        full_name = f'{layer_s}/token/{sample_idx_s}/{name}'
        for i in range(T):
            tb_writer.add_histogram(full_name, tensor[0, i], global_step=i, **hist_kw)
