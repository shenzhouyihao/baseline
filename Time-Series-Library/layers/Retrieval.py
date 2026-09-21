import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm
        
        self.with_dec = with_dec
        self.return_key = return_key
        
    def prepare_dataset(self, train_data):
        # Process in chunks to avoid massive temporary allocations
        chunk_size = 1000
        n_samples = len(train_data)

        # Get shapes from first sample
        td0 = train_data[0]
        if self.with_dec:
            y_shape = td0[1][-(train_data.pred_len + train_data.label_len):].shape
        else:
            y_shape = td0[1][-train_data.pred_len:].shape

        train_shape = td0[0].shape

        # Pre-allocate output tensors in float16
        train_data_all = torch.zeros((n_samples, *train_shape), dtype=torch.float16)
        y_data_all = torch.zeros((n_samples, *y_shape), dtype=torch.float16)

        # Fill in chunks to avoid large temporary arrays
        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)

            # Build chunk arrays
            train_chunk = []
            y_chunk = []

            for i in range(chunk_start, chunk_end):
                td = train_data[i]
                train_chunk.append(td[0])

                if self.with_dec:
                    y_chunk.append(td[1][-(train_data.pred_len + train_data.label_len):])
                else:
                    y_chunk.append(td[1][-train_data.pred_len:])

            # Stack chunk and convert to float16
            train_data_all[chunk_start:chunk_end] = torch.from_numpy(
                np.stack(train_chunk, axis=0).astype(np.float16)
            )
            y_data_all[chunk_start:chunk_end] = torch.from_numpy(
                np.stack(y_chunk, axis=0).astype(np.float16)
            )

            del train_chunk, y_chunk

        # Decompose train_data and store in float16 on CPU
        self.train_data_all_mg, _ = self.decompose_mg(train_data_all.float())
        self.train_data_all_mg = self.train_data_all_mg.half()
        del train_data_all

        # Store y_data as original (not decomposed) to save memory
        self.y_data_all = y_data_all

        import gc
        gc.collect()

        self.n_train = self.train_data_all_mg.shape[1]

    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # T, S, C

        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            
            mg.append(cur)
#             data_all = data_all - cur
            
        mg = torch.stack(mg, dim=0) # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:,-1:,:]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None
            
        offset = torch.stack(offset, dim=0)
            
        return mg, offset
    
    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape
        
        bx = key - torch.mean(key, dim=2, keepdim=True)
        
        iters = math.ceil(train_len / in_bsz)
        
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            
            cur_data = data_all[:, start_idx:end_idx].to(key.device).float()
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2)
        
        return sim
        
    def retrieve(self, x, index, train=True):
        index = index.to(x.device)

        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)

        # Decompose input on-the-fly
        x_mg, mg_offset = self.decompose_mg(x) # G, B, S, C

        # Use pre-computed decomposed training data from CPU (float16)
        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C (on CPU, float16)
            x_mg.flatten(start_dim=2), # G, B, S * C
        ) # G, B, T
            
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
            
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
            
            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

        sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T
                
        topm_index = torch.topk(sim, self.topm, dim=1).indices
        ranking_sim = torch.ones_like(sim) * float('-inf')
        
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]
        
        sim = sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train) # G, B, T

        data_len, seq_len, channels = self.train_data_all_mg.shape[1:]

        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu() # G, B, T

        # Extract top-k indices
        top_k_indices = topm_index.reshape(self.n_period, bsz, self.topm).cpu() # G, B, K

        # Process each period group separately (each may have different top-k samples)
        pred_components = []
        for g in range(self.n_period):
            k_idx_g = top_k_indices[g]  # B, K (cpu, long)

            # Gather top-k original y_data samples for this period's queries
            k_indices_flat = k_idx_g.flatten()  # B*K
            y_topk_orig = self.y_data_all[k_indices_flat].to(x.device).float()  # B*K, P, C

            # Decompose top-k samples on GPU
            y_topk_decomp, _ = self.decompose_mg(y_topk_orig)  # G', B*K, P, C

            # Extract this period's decomposition component
            y_topk_g_flat = y_topk_decomp[g].flatten(start_dim=1)  # B*K, P*C
            y_topk_g = y_topk_g_flat.reshape(bsz, self.topm, -1)  # B, K, P*C

            # Compute weighted sum
            prob_topk_g = torch.gather(ranking_prob[g], 1, k_idx_g).to(x.device)  # B, K
            pred_g = (prob_topk_g.unsqueeze(-1) * y_topk_g).sum(dim=1)  # B, P*C
            pred_components.append(pred_g)

        pred_from_retrieval = torch.stack(pred_components, dim=0).reshape(self.n_period, bsz, -1, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)

        return pred_from_retrieval
    
    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        assert(self.train_data_all_mg != None)
        
        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        
        retrievals = []
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(rt_loader)):
                # Generate indices for this batch
                batch_size = batch_x.shape[0]
                start_idx = batch_idx * rt_loader.batch_size
                index = torch.arange(start_idx, start_idx + batch_size)
                pred_from_retrieval = self.retrieve(batch_x.float().to(device), index, train=train)
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                
        retrievals = torch.cat(retrievals, dim=1)
        
        return retrievals