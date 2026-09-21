import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import RetrievalTool

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
#         self.decompsition = series_decomp(configs.moving_avg)
#         self.individual = individual
        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        
        self.n_period = configs.n_period
        self.topm = configs.topm
        
        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
        )
        
        self.period_num = self.rt.period_num[-1 * self.n_period:]
        
        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

#         if self.task_name == 'classification':
#             self.projection = nn.Linear(
#                 configs.enc_in * configs.seq_len, configs.num_class)

    def prepare_dataset(self, train_data, valid_data, test_data):
        print('🔍 Preparing RAFT retrieval tool (on-demand mode to save memory)...')
        self.rt.prepare_dataset(train_data)
        print('✅ RAFT retrieval tool ready (will compute retrieval on-the-fly per batch)')

    def encoder(self, x, index, mode):
        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1) # B, P, C

        # Compute retrieval on-the-fly for this batch to save memory
        train_mode = (mode == 'train')
        pred_from_retrieval = self.rt.retrieve(x, index, train=train_mode) # G, B, P, C
        
        retrieval_pred_list = []
        
        # Compress repeating dimensions
        for i, pr in enumerate(pred_from_retrieval):
            assert((bsz, self.pred_len, channels) == pr.shape)
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]
            
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)
            
            retrieval_pred_list.append(pr)

        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)
        retrieval_pred_list = retrieval_pred_list.sum(dim=1)
        
        pred = torch.cat([x_pred_from_x, retrieval_pred_list], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)
        
        pred = pred + x_offset
        
        return pred

    def forecast(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def imputation(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        # Encoder
        enc_out = self.encoder(x_enc, index, mode)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, batch_index=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            batch_size = x_enc.shape[0]
            if batch_index is None:
                index = torch.arange(batch_size, device=x_enc.device)
            else:
                index = batch_index
            mode = 'train' if self.training else 'test'
            dec_out = self.forecast(x_enc, index, mode)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            batch_size = x_enc.shape[0]
            if batch_index is None:
                index = torch.arange(batch_size, device=x_enc.device)
            else:
                index = batch_index
            mode = 'train' if self.training else 'test'
            dec_out = self.imputation(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            batch_size = x_enc.shape[0]
            if batch_index is None:
                index = torch.arange(batch_size, device=x_enc.device)
            else:
                index = batch_index
            mode = 'train' if self.training else 'test'
            dec_out = self.anomaly_detection(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            batch_size = x_enc.shape[0]
            if batch_index is None:
                index = torch.arange(batch_size, device=x_enc.device)
            else:
                index = batch_index
            mode = 'train' if self.training else 'test'
            dec_out = self.classification(x_enc, index, mode)
            return dec_out  # [B, N]
        return None
