import logging
import random
import math

## numpy
import numpy as np

## PyTorch
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import torch.nn.functional as F

import wandb

from torch.utils.data import DataLoader

from utils.utils import analyze_spikes, analyse_trend, visualize, generate_square_subsequent_mask, create_causal_memory_mask, create_causal_memory_mask_new
from data_provider.data_loader import Dataset_EEGEyeNet, create_custom_collate_fn, create_custom_collate_fn_feat_norm, create_custom_collate_fn_normalized

logger = logging.getLogger(__name__)


class ExpMain:
    def __init__(self, args, model):
        self.args = args
        self.model = model.to(self.args.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        self.scaler = torch.GradScaler("cuda")

    def train(self):
        # initialize dataset
        train_dataset = Dataset_EEGEyeNet(args=self.args, flag="train")
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                                      collate_fn=create_custom_collate_fn(self.args))
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        self.optimizer, 
                        max_lr=self.args.learning_rate, 
                        steps_per_epoch=len(train_dataloader),
                        epochs=self.args.epochs,
                        div_factor=25,
                        final_div_factor=25,
                        pct_start=0.3
                    )
        
        val_dataset = Dataset_EEGEyeNet(args=self.args, flag="val")
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True,
                                      collate_fn=create_custom_collate_fn(self.args))

        test_dataset = Dataset_EEGEyeNet(args=self.args, flag="test")
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, 
                                     collate_fn=create_custom_collate_fn(self.args))
                   
        for epoch in range(1, self.args.epochs + 1):
            # print(f"EPOCH: {epoch}")
            self._this_epoch = epoch
            logger.info(f"Starting Training Epoch {epoch}")
            train_loss = self.train_one_epoch(train_dataloader)            
            logger.info(f'Training epoch {epoch}/{self.args.epochs} \t\t Training Loss: {train_loss}')

            val_loss = self.val(val_dataloader)
            logger.info(f'Validation epoch {epoch}/{self.args.epochs} \t\t Validation Loss: {val_loss}')
            
            test_loss = self.test(test_dataloader)
            logger.info(f'Testing epoch {epoch}/{self.args.epochs} \t\t Testing Loss: {test_loss}')
            # torch.save(self.model.state_dict(), f"{self.args.models_dir}/model_{epoch}.pth")
            
    def train_one_epoch(self, train_dataloader):
        # torch.autograd.set_detect_anomaly(True)

        self.model.train()
        train_mse = 0
        train_mae = 0
        train_rmse = 0
        total_true_spikes, total_pred_spikes, TPs, FPs, FNs, TNs = 0, 0, 0, 0, 0, 0        
        trend_MSEs = []

        # declare ratio to implement scheduled sampling to mitigate exposure bias
        teacher_forcing_ratio = max(0.0, 1.0 - (self._this_epoch / (self.args.epochs * 0.7)))
        # teacher_forcing_ratio = 0.5
        logger.info(f"Starting Training Epoch {self._this_epoch}: TFR: {teacher_forcing_ratio:.4f}")

        # attn_mask = sliding_causal_mask(seq_len=self.args.max_len, window=self.args.context_window, device=self.args.device) 
        tgt_attn_mask = generate_square_subsequent_mask(sz=self.args.context_window).to(self.args.device)
        if self.args.model == "new":
            src_attn_mask = generate_square_subsequent_mask(sz=int((self.args.context_window * self.args.metric_window_duration / self.args.feature_window_duration) / self.args.time_patch_len)).to(self.args.device)
            mem_attn_mask = create_causal_memory_mask_new(T=self.args.context_window,
                                                          num_feat_per_metric=self.args.num_features_per_metric,
                                                          time_patch_len=self.args.time_patch_len,
                                                          device=self.args.device)
        else:
            src_attn_mask = generate_square_subsequent_mask(sz=self.args.context_window * self.args.num_features_per_metric).to(self.args.device)
            mem_attn_mask = create_causal_memory_mask(T=self.args.context_window, num_feat_per_metric=self.args.num_features_per_metric, device=self.args.device)

        for i, data in enumerate(train_dataloader):
            gaze_x = data["gaze_x"].to(self.args.device)
            gaze_y = data["gaze_y"].to(self.args.device)
            gaze_vel = data["gaze_vel"].to(self.args.device)
            gaze_acc = data["gaze_acc"].to(self.args.device)
            pupil = data["pupil"].to(self.args.device)
            stimulus = data["stimulus"].to(self.args.device)
            labels = data["labels"].to(self.args.device)
            masks = data["loss_masks"].to(self.args.device)

            # print(f"labels: {labels}")
            # print(f"gaze X: {gaze_x}")

            B, S = labels.shape
            self.optimizer.zero_grad()
        
            # key_padding_mask = padding_mask(seq_lengths=seq_lens, max_len=self.args.max_len, device=self.args.device)
            # attn_mask = generate_batch_safe_mask(seq_len=self.args.max_len, window=self.args.context_window, pad_mask=key_padding_mask, num_heads=self.args.num_heads, device=self.args.device)

            # print(f"Key Padding Mask: {key_padding_mask}")
            # print(f"SRC Attention Mask: {src_attn_mask}")
            # print(f"MEM Attention Mask: {mem_attn_mask}")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = labels[:, :self.args.context_window]
                all_predictions = [outputs]
            
                for t in range(self.args.context_window, S):
                    # print(f"outputs: {outputs}")
                    # print(f"gaze_x: {gaze_x[:, t - self.args.context_window: t]}")
                    # print(f"mask {attn_mask[:, :t].shape}")

                    new_pred = self.model(t - self.args.context_window,
                        gaze_x[:, t - self.args.context_window: t],
                        gaze_y[:, t - self.args.context_window: t],
                        gaze_vel[:, t - self.args.context_window: t],
                        gaze_acc[:, t - self.args.context_window: t],
                        pupil[:, t - self.args.context_window: t],
                        stimulus[:, t - self.args.context_window: t],
                        outputs,  # Use outputs generated so far
                        src_mask=src_attn_mask,
                        tgt_mask=tgt_attn_mask,
                        memory_mask=mem_attn_mask,
                        src_key_padding_mask=None,
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=None
                    )[:, -1:]

                    # print(f"new_pred: {new_pred}")
                    all_predictions.append(new_pred)               # store current output

                    if random.random() < teacher_forcing_ratio:
                        next_input = labels[:, t:t+1]     
                        # print("TF")               
                    else:                        
                        next_input = new_pred
                        # print("A")               
                    
                    # print(f"next_input: {next_input}")
                    # Update outputs with the predicted token
                    outputs = torch.cat([outputs[:, 1:], next_input], dim=1)
    

                predictions = torch.cat(all_predictions, dim=1)

                preds = torch.masked_select(predictions, masks)
                trues = torch.masked_select(labels, masks)

                # print(f"Masked Ground truths: {trues}")
                # print(f"Masked Predictions: {preds}")
                
                loss = self.criterion(preds, trues)
            
            self.scaler.scale(loss).backward()         

            # total_norm = 0
            # for name, p in self.model.named_parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2).item()
            #         total_norm += param_norm ** 2
            #         print(f"{name} grad norm: {param_norm}")

            # # Optional: Gradient clipping
            # self.scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            
            # print(f"preds: {preds}")
            tts, tps, tp, fp, fn, tn = analyze_spikes(predictions, labels, masks)       
            trend_mse = analyse_trend(predictions, labels, masks)
            trend_MSEs.extend(trend_mse)

            # print(f"Ground truths: {labels}")
            # print(f"Predictions: {preds}")

            if i == 0 and self._this_epoch % 10 == 1:  # For batches 0, 1, 2, 3
                visualize(self._this_epoch, self.args.plots_dir_train, predictions, labels, self.args.max_len,
                    title=f"Train Epoch {self._this_epoch}"
                    )


            total_true_spikes += tts
            total_pred_spikes += tps
            TPs += tp
            FPs += fp
            FNs += fn
            TNs += tn
            train_mse += loss.item()
            train_mae += F.l1_loss(preds, trues, reduction='mean').item()
        
        train_mse = train_mse / len(train_dataloader)
        train_mae = train_mae / len(train_dataloader)
        train_rmse = math.sqrt(train_mse)   

        wandb.log({
            "train/epoch_mse": train_mse ,  
            "train/epoch_mae": train_mae ,
            "train/epoch_rmse": train_rmse ,
            "train/teacher_forcing_ratio": teacher_forcing_ratio,
            "train/epoch": self._this_epoch,
            "train/total_true_spikes": total_true_spikes,
            "train/total_pred_spikes": total_pred_spikes,
            "train/true_positives": TPs,
            "train/false_positives": FPs,
            "train/false_negatives": FNs,
            "train/true_negatives": TNs,
            "train/trend_MSE": np.mean(trend_MSEs),
        })
        return train_mse

    def val(self, val_dataloader):   
        self.model.eval()
        # print("VALIDATION")
        
        val_mse = 0.0
        val_mae = 0.0
        val_rmse = 0.0
        total_true_spikes, total_pred_spikes, TPs, FPs, FNs, TNs = 0, 0, 0, 0, 0, 0
        trend_MSEs = []

        # attn_mask = sliding_causal_mask(seq_len=self.args.max_len, window=self.args.context_window, device=self.args.device)  
        tgt_attn_mask = generate_square_subsequent_mask(sz=self.args.context_window).to(self.args.device)
        if self.args.model == "new":
            src_attn_mask = generate_square_subsequent_mask(sz=int((self.args.context_window * self.args.metric_window_duration / self.args.feature_window_duration) / self.args.time_patch_len)).to(self.args.device)
            mem_attn_mask = create_causal_memory_mask_new(T=self.args.context_window,
                                                          num_feat_per_metric=self.args.num_features_per_metric,
                                                          time_patch_len=self.args.time_patch_len,
                                                          device=self.args.device)
        else:
            src_attn_mask = generate_square_subsequent_mask(sz=self.args.context_window * self.args.num_features_per_metric).to(self.args.device)
            mem_attn_mask = create_causal_memory_mask(T=self.args.context_window, num_feat_per_metric=self.args.num_features_per_metric, device=self.args.device)

        with torch.no_grad():
            for i, data in enumerate(val_dataloader):   
                gaze_x = data["gaze_x"].to(self.args.device)
                gaze_y = data["gaze_y"].to(self.args.device)
                gaze_vel = data["gaze_vel"].to(self.args.device)
                gaze_acc = data["gaze_acc"].to(self.args.device)
                pupil = data["pupil"].to(self.args.device)
                stimulus = data["stimulus"].to(self.args.device)
                labels = data["labels"].to(self.args.device)
                masks = data["loss_masks"].to(self.args.device)

                # print(f"Seq Lengths: {seq_lens}")

                # print(f"Ground truths: {labels}")

                B, S = labels.shape
                # key_padding_mask = padding_mask(seq_lengths=seq_lens, max_len=self.args.max_len, device=self.args.device)                
                # attn_mask = generate_batch_safe_mask(seq_len=self.args.max_len, window=self.args.context_window, pad_mask=key_padding_mask, num_heads=self.args.num_heads, device=self.args.device)

                # print(f"Key Padding Mask: {key_padding_mask}")
                # print(f"Attention Mask: {attn_mask}")
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = labels[:, :self.args.context_window]
                    all_predictions = [outputs]
                
                    for t in range(self.args.context_window, S):
                        # print(f"{outputs}")
                        
                        new_pred = self.model(t - self.args.context_window,
                            gaze_x[:, t - self.args.context_window: t],
                            gaze_y[:, t - self.args.context_window: t],
                            gaze_vel[:, t - self.args.context_window: t],
                            gaze_acc[:, t - self.args.context_window: t],
                            pupil[:, t - self.args.context_window: t],
                            stimulus[:, t - self.args.context_window: t],
                            outputs,  # Use outputs generated so far
                            src_mask=src_attn_mask,
                            tgt_mask=tgt_attn_mask,
                            memory_mask=mem_attn_mask,
                            src_key_padding_mask=None,
                            tgt_key_padding_mask=None,
                            memory_key_padding_mask=None
                        )[:, -1:]

                        # print(f"new_pred: {new_pred}")

                        all_predictions.append(new_pred)
                        
                        # Update outputs with the predicted token
                        outputs = torch.cat([outputs[:, 1:], new_pred], dim=1)
                    # print(f"outputs: {outputs}")
                    # print(f"Outputs: {outputs}")
                    # print(f"Labels: {labels}")
                    # print(f"Masks: {masks}")(f"Predictions: {predicted_br}")
                    
                    preds = torch.cat(all_predictions, dim=1)
                    predicted_br = torch.masked_select(preds, masks)
                    true_br = torch.masked_select(labels, masks)

                    # print(f"Ground truths: {labels}")
                    # print(f"Predictions: {preds}")
                    
                    # print(f"predicted_br: {predicted_br}")
                    # print(f"true_br: {true_br}")

                    loss = self.criterion(predicted_br, true_br)

                tts, tps, tp, fp, fn, tn = analyze_spikes(preds, labels, masks)
                trend_mse = analyse_trend(preds, labels, masks)
                trend_MSEs.extend(trend_mse)
                
                if i == 0 and self._this_epoch % 10 == 1:  # For batches 0, 1, 2, 3
                    visualize(self._this_epoch, self.args.plots_dir_val, preds, labels, self.args.max_len,
                        title=f"Val Epoch {self._this_epoch}"
                        )

                total_true_spikes += tts
                total_pred_spikes += tps
                TPs += tp
                FPs += fp
                FNs += fn
                TNs += tn
                val_mse += loss.item()
                val_mae += F.l1_loss(predicted_br, true_br, reduction='mean').item()
        
        val_mse = val_mse / len(val_dataloader)
        val_mae = val_mae / len(val_dataloader)
        val_rmse = math.sqrt(val_mse)   
                

        wandb.log({
            "val/epoch_mse": val_mse,
            "val/epoch_mae": val_mae,
            "val/epoch_rmse": val_rmse,
            "val/epoch": self._this_epoch,
            "val/total_true_spikes": total_true_spikes,
            "val/total_pred_spikes": total_pred_spikes,
            "val/true_positives": TPs,
            "val/false_positives": FPs,
            "val/false_negatives": FNs,
            "val/true_negatives": TNs,
            "val/trend_MSE": np.mean(trend_MSEs),
        })
        return val_mse
            
    def test(self, test_dataloader):   
        self.model.eval()
        # print("TESTING")

        test_mse = 0.0
        test_mae = 0.0
        test_rmse = 0.0
        total_true_spikes, total_pred_spikes, TPs, FPs, FNs, TNs = 0, 0, 0, 0, 0, 0
        trend_MSEs = []

        # attn_mask = sliding_causal_mask(seq_len=self.args.max_len, window=self.args.context_window, device=self.args.device)  
        tgt_attn_mask = generate_square_subsequent_mask(sz=self.args.context_window).to(self.args.device)
        if self.args.model == "new":
            src_attn_mask = generate_square_subsequent_mask(sz=int((self.args.context_window * self.args.metric_window_duration / self.args.feature_window_duration) / self.args.time_patch_len)).to(self.args.device)
            mem_attn_mask = create_causal_memory_mask_new(T=self.args.context_window,
                                                          num_feat_per_metric=self.args.num_features_per_metric,
                                                          time_patch_len=self.args.time_patch_len,
                                                          device=self.args.device)
        else:
            src_attn_mask = generate_square_subsequent_mask(sz=self.args.context_window * self.args.num_features_per_metric).to(self.args.device)
            mem_attn_mask = create_causal_memory_mask(T=self.args.context_window, num_feat_per_metric=self.args.num_features_per_metric, device=self.args.device)

        with torch.no_grad():
            for i, data in enumerate(test_dataloader):   
                gaze_x = data["gaze_x"].to(self.args.device)
                gaze_y = data["gaze_y"].to(self.args.device)
                gaze_vel = data["gaze_vel"].to(self.args.device)
                gaze_acc = data["gaze_acc"].to(self.args.device)
                pupil = data["pupil"].to(self.args.device)
                stimulus = data["stimulus"].to(self.args.device)
                labels = data["labels"].to(self.args.device)
                masks = data["loss_masks"].to(self.args.device)

                # print(f"Ground truths: {labels}")

                B, S = labels.shape
                # key_padding_mask = padding_mask(seq_lengths=seq_lens, max_len=self.args.max_len, device=self.args.device)
                # attn_mask = generate_batch_safe_mask(seq_len=self.args.max_len, window=self.args.context_window, pad_mask=key_padding_mask, num_heads=self.args.num_heads, device=self.args.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = labels[:, :self.args.context_window]
                    all_predictions = [outputs]

                    for t in range(self.args.context_window, S):
                        # print(f"{outputs}")

                        new_pred = self.model(t - self.args.context_window,
                            gaze_x[:, t - self.args.context_window: t],
                            gaze_y[:, t - self.args.context_window: t],
                            gaze_vel[:, t - self.args.context_window: t],
                            gaze_acc[:, t - self.args.context_window: t],
                            pupil[:, t - self.args.context_window: t],
                            stimulus[:, t - self.args.context_window: t],
                            outputs,  # Use outputs generated so far
                            src_mask=src_attn_mask,
                            tgt_mask=tgt_attn_mask,
                            memory_mask=mem_attn_mask,
                            src_key_padding_mask=None,
                            tgt_key_padding_mask=None,
                            memory_key_padding_mask=None
                        )[:, -1:]

                        # print(f"new_pred: {new_pred}")

                        all_predictions.append(new_pred)
                        
                        # Update outputs with the predicted token                    
                        outputs = torch.cat([outputs[:, 1:], new_pred], dim=1)
                        # print(f"outputs: {outputs}")
                        # print(f"Outputs: {outputs}")
                        # print(f"Labels: {labels}")
                        # print(f"Masks: {masks}")(f"Predictions: {predicted_br}")                   
                
                    preds = torch.cat(all_predictions, dim=1)

                    # print(f"Ground truths: {labels}")
                    # print(f"Predictions: {preds}")

                    # print(f"predicted_br: {predicted_br}")
                    predicted_br = torch.masked_select(preds, masks)
                    true_br = torch.masked_select(labels, masks)

                    loss = self.criterion(predicted_br, true_br)

                tts, tps, tp, fp, fn, tn = analyze_spikes(preds, labels, masks)
                trend_mse = analyse_trend(preds, labels, masks)
                trend_MSEs.extend(trend_mse)
                
                if i == 0 and self._this_epoch % 10 == 1:  # For batches 0, 1, 2, 3
                    visualize(self._this_epoch, self.args.plots_dir_test, preds, labels, self.args.max_len,
                        title=f"Test Epoch {self._this_epoch}"
                        )
                total_true_spikes += tts
                total_pred_spikes += tps
                TPs += tp
                FPs += fp
                FNs += fn
                TNs += tn
                test_mse += loss.item()
                test_mae += F.l1_loss(predicted_br, true_br, reduction='mean').item()
        
        test_mse = test_mse / len(test_dataloader)
        test_mae = test_mae / len(test_dataloader)
        test_rmse = math.sqrt(test_mse)   

        wandb.log({
            "test/epoch_mse": test_mse,
            "test/epoch_mae": test_mae,
            "test/epoch_rmse": test_rmse,
            "test/epoch": self._this_epoch,
            "test/total_true_spikes": total_true_spikes,
            "test/total_pred_spikes": total_pred_spikes,
            "test/true_positives": TPs,
            "test/false_positives": FPs,
            "test/false_negatives": FNs,
            "test/true_negatives": TNs,
            "test/trend_MSE": np.mean(trend_MSEs),
        })
        return test_mse