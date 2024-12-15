import torch
from typing import Tuple, Callable
import logging

import os

os.environ['CUDA_LAUNCH_BLOCKING']='1'

def do_nothing(x: torch.Tensor, mode: str = None, prune: bool = None, unmerge_mode = None):
    """
    A versatile placeholder...
    """
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def deprecated_bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     tome_info: dict,
                                     no_rand: bool = False,
                                     unmerge_mode: str = 'token_merge',
                                     cache: any = None,
                                     rand_indices: list = None,
                                     generator: torch.Generator = None
                                     ) -> Tuple[Callable, Callable]:
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    # # Force stop (In most case, broadcast_start < cache_start
    if cache.step < tome_info['args']['merge_start'] or cache.step > tome_info['args']['merge_end']:
        if cache.step == tome_info['args']['merge_start'] - 1 and unmerge_mode == 'cache_merge':
            def initial_push(x: torch.Tensor):
                        cache.push(x)
                        return x
            return do_nothing, initial_push
        else:
            return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            if unmerge_mode == 'cache_merge':
                # retrieve from a pre-defined semi-random schedule
                rand_idx = rand_indices.pop().to(generator.device)
            else:
                rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx # this assumes we only choose one dst from a grid
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        spatial_scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = spatial_scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)  # dst indices src tokens should merge to

    def merge(x: torch.Tensor, mode="mean", prune=False) -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))

        if not prune:
            src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce_(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        # Simply concat
        out = torch.cat([unm, dst], dim=1)
        print(f"\033[96mMerge\033[0m: feature map merged from \033[95m{x.shape}\033[0m to \033[95m{out.shape}\033[0m "
                      f"at block index: \033[91m{cache.index}\033[0m")
        return out

    def unmerge(x: torch.Tensor, unmerge_mode=unmerge_mode) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        if unmerge_mode == 'cache_merge' and tome_info['args']['cache_start'] <= cache.step <= tome_info['args']['cache_end']:
            # == Branch 1: Improved Merging middle steps
            cache.push(dst, index=b_idx.expand(B, num_dst, c))
            if tome_info['args']['push_unmerged']:
                cache.push(unm, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c))
            src = cache.pop(index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c))

        else:
            # == Branch 2: Token Merging & Improved Merging first/last steps
            # Proceed vanilla token unmerging, while updating the cache if cache unmerging is enabled
            if unmerge_mode == 'cache_merge' and cache.feature_map is not None:
                cache.push(dst, index=b_idx.expand(B, num_dst, c))
                if tome_info['args']['push_unmerged']:
                    cache.push(unm, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c))

            src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # == Combine back to the original shape (Branch 1 SubBranch 2 & Branch 2)
        test_idx = gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c)
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        # For the first step
        if unmerge_mode == 'cache_merge' and cache.feature_map is None:
            cache.push(out)

        return out

    return merge, unmerge


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     tome_info: dict,
                                     no_rand: bool = False,
                                     unmerge_mode: str = 'token_merge',
                                     cache: any = None,
                                     rand_indices: list = None,
                                     generator: torch.Generator = None
                                     ) -> Tuple[Callable, Callable]:
    """
    Core algorithm - V2
    """

    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    if cache.step < tome_info['args']['merge_start'] or cache.step > tome_info['args']['merge_end']:
        if cache.step == tome_info['args']['merge_start'] - 1 and unmerge_mode == 'cache_merge':
            def initial_push(x: torch.Tensor):
                cache.push(x)
                return x

            return do_nothing, initial_push
        else:
            return do_nothing, do_nothing


    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx
        h_crop, w_crop = hsy * sy, wsx * sx

        # Generate random indices for destination tokens within each kernel
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, dtype=torch.int32, device=metric.device)
        else:
            rand_idx = (rand_indices.pop().to(metric.device) if unmerge_mode == 'cache_merge'
                else torch.randint(sy * sx, (hsy, wsx, 1), generator=generator, device=metric.device)
            )

        # Create a grid of indices representing the image pixels
        idx_grid = torch.arange(h_crop * w_crop, device=metric.device).reshape(h_crop, w_crop)

        # Reshape indices to match kernel blocks
        idx_flat = idx_grid.reshape(hsy, sy, wsx, sx).permute(0, 2, 1, 3).reshape(hsy, wsx, sy * sx)

        # Identify destination indices and mark them in idx_flat
        b_idx = idx_flat.gather(2, rand_idx).view(-1)
        idx_flat.scatter_(2, rand_idx, -1)

        # Extract source indices directly from idx_flat without creating a mask
        a_idx = idx_flat[idx_flat != -1]

        # Handle extra pixels if the image dimensions are not divisible by sy or sx
        if h_crop < h:
            extra_rows = idx_grid[h_crop:, :w_crop].flatten()
            src_idx = torch.cat((a_idx, extra_rows), dim=0)
        if w_crop < w:
            extra_cols = idx_grid[:h_crop, w_crop:].flatten()
            src_idx = torch.cat((a_idx, extra_cols), dim=0)
        if h_crop < h and w_crop < w:
            corner_pixels = idx_grid[h_crop:, w_crop:].flatten()
            src_idx = torch.cat((a_idx, corner_pixels), dim=0)

        a_idx = a_idx.view(1, -1, 1) # src
        b_idx = b_idx.view(1, -1, 1) # dst

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx # this assumes we only choose one dst from a grid
        temporal_score = cache.cache_bus.temporal_score.clone().reshape(1, -1, 1)

        # # todo: untested 1
        # Flatten the tensors for efficient computation
        a_idx_flat = a_idx.view(-1)  # Shape: [6912]
        b_idx_flat = b_idx.view(-1)  # Shape: [2304]
        score_flat = temporal_score.view(-1)  # Shape: [9216]

        # Create a boolean mask for indices in b_idx_flat
        b_mask = torch.zeros(score_flat.size(0), dtype=torch.bool, device=temporal_score.device)
        b_mask[b_idx_flat] = True  # Mark indices present in b_idx_flat

        # Identify indices in a_idx_flat to move to b_idx_flat
        move_mask = (score_flat[a_idx_flat] == 1) & (~b_mask[a_idx_flat])
        indices_to_move = a_idx_flat[move_mask]

        # Update a_idx_flat by removing indices_to_move
        a_idx_flat = a_idx_flat[~move_mask]

        # Update b_idx_flat by adding indices_to_move
        b_idx_flat = torch.cat([b_idx_flat, indices_to_move])

        # Reshape back to original dimensions
        a_idx = a_idx_flat.view(1, -1, 1)  # Updated a_idx
        b_idx = b_idx_flat.view(1, -1, 1)  # Updated b_idx

        # Update r and num_dst based on new shapes
        r = a_idx.size(1)
        num_dst = b_idx.size(1)

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        src_idx = torch.arange(r, device=metric.device).view(1, -1, 1)  # Indices in a_idx
        # dst_idx = torch.randint(0, num_dst, (1, r, 1), device=metric.device)  # Random indices in b_idx
        unm_idx = torch.tensor([], device=metric.device, dtype=torch.long).view(1, 0, 1)  # No unmerged tokens


    def prune(x: torch.Tensor, prune=False) -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))

        # Simply concat
        out = torch.cat([unm, dst], dim=1)
        print(f"\033[96mMerge\033[0m: feature map merged from \033[95m{x.shape}\033[0m to \033[95m{out.shape}\033[0m "f"at block index: \033[91m{cache.index}\033[0m")
        return out

    def reconstruct(x: torch.Tensor, unmerge_mode=unmerge_mode) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        if unmerge_mode == 'cache_merge' and tome_info['args']['cache_start'] <= cache.step <= tome_info['args']['cache_end']:
            cache.push(dst, index=b_idx.expand(B, num_dst, c))
            src = cache.pop(index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c))
        else:
            raise RuntimeError

        # == Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return prune, reconstruct