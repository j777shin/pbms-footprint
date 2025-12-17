import logging
import random
import time
from typing import Tuple, Union
import os
import numpy as np
import torch
import tqdm
from py_oneliner import one_liner

from building_footprint_segmentation.helpers.callbacks import SchedulerCallback
from building_footprint_segmentation.utils.operations import (
    handle_dictionary,
    compute_eta,
    dict_to_string,
)
from building_footprint_segmentation.utils.py_network import gpu_variable, extract_state

logger = logging.getLogger("segmentation")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from collections import defaultdict

class Trainer:
    def __init__(
        self, model, criterion, optimizer, loader, metrics, callbacks, scheduler
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loader = loader
        self.metrics = metrics
        self.callbacks = callbacks

        if scheduler is not None:
            self.callbacks.append(SchedulerCallback(scheduler))

    def previous_state(self, state):
        assert {
            "model",
            "optimizer",
            "step",
            "start_epoch",
            "end_epoch",
            "bst_vld_loss",
        } == set(list(state.keys())), (
            "Expected 'state' to have ['model', 'optimizer', 'step', 'start_epoch', 'end_epoch', 'bst_vld_loss']'"
            "got %s",
            (list(state.keys()),),
        )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        return (
            state["step"],
            state["start_epoch"],
            state["end_epoch"],
            state["bst_vld_loss"],
        )

    def resume(self, state: str, new_end_epoch=None):
        step, start_epoch, end_epoch, bst_vld_loss = self.previous_state(
            extract_state(state)
        )
        if new_end_epoch is not None:
            logger.debug(f"Overriding end epoch from {end_epoch} to {new_end_epoch}")
            end_epoch = new_end_epoch

        logger.debug(
            f"Resuming State {state}, with, step: {step}, start_epoch: {start_epoch}"
        )
        self.train(start_epoch, end_epoch, step, bst_vld_loss)

    def train(self, start_epoch, end_epoch, step: int = 0, bst_vld_loss: float = None):
        logger.debug(f"Training Begin")
        self.callbacks.on_begin()
        for ongoing_epoch in range(start_epoch, end_epoch):
            epoch_logs = dict()
            random.seed()

            lr = self.optimizer.param_groups[0]["lr"]

            progress_bar = tqdm.tqdm(
                total=(
                    len(self.loader.train_loader) * self.loader.train_loader.batch_size
                )
            )
            progress_bar.set_description("Epoch {}, lr {}".format(ongoing_epoch, lr))

            try:
                logger.debug("Setting Learning rate : {}".format(lr))
                epoch_logs = handle_dictionary(epoch_logs, "lr", lr)

                if not self.model.training:
                    self.model.train()

                train_loss, train_metric, step, progress_bar = self.state_train(
                    step, progress_bar
                )
                progress_bar.close()

                self.model.eval()
                valid_loss, valid_metric = self.state_validate()

                epoch_logs = handle_dictionary(epoch_logs, "train_loss", train_loss)
                epoch_logs = handle_dictionary(epoch_logs, "valid_loss", valid_loss)

                epoch_logs = handle_dictionary(epoch_logs, "train_metric", train_metric)
                epoch_logs = handle_dictionary(epoch_logs, "valid_metric", valid_metric)

                if (bst_vld_loss is None) or (valid_loss < bst_vld_loss):
                    bst_vld_loss = valid_loss

                epoch_logs = handle_dictionary(epoch_logs, "model", self.model)
                epoch_logs = handle_dictionary(
                    epoch_logs, "test_loader", self.loader.test_loader
                )

                self.callbacks.on_epoch_end(
                    ongoing_epoch,
                    logs={
                        **epoch_logs,
                        **self.collect_state(
                            ongoing_epoch, end_epoch, step, bst_vld_loss, "complete"
                        ),
                    },
                )

                logger.debug(
                    "Train Loss {}, Valid Loss {}".format(train_loss, valid_loss)
                )
                logger.debug("Train Metric {}".format(train_metric))
                logger.debug("Valid Metric {}".format(valid_metric))

                one_liner.one_line(
                    tag="Loss",
                    tag_data=f"train: {train_loss}, validation: {valid_loss}",
                    tag_color="cyan",
                    to_reset_data=True,
                )

                one_liner.one_line(
                    tag="Train Metric",
                    tag_data=dict_to_string(train_metric),
                    tag_color="cyan",
                    to_reset_data=True,
                    to_new_line_data=True,
                )
                one_liner.one_line(
                    tag="Valid Metric",
                    tag_data=dict_to_string(valid_metric),
                    tag_color="cyan",
                    to_new_line_data=True,
                    to_reset_data=True,
                )

            except KeyboardInterrupt:
                progress_bar.close()
                self.callbacks.interruption(
                    logs={
                        **epoch_logs,
                        **self.collect_state(
                            ongoing_epoch, end_epoch, step, bst_vld_loss, "interruption"
                        ),
                    }
                )

                one_liner.one_line(
                    tag="KeyBoard Interrupt",
                    tag_data=f"State Saved at epoch {ongoing_epoch}",
                    tag_color="cyan",
                    to_reset_data=True,
                )
                raise KeyboardInterrupt
            except Exception as ex:
                progress_bar.close()
                one_liner.one_line(
                    tag="Exception",
                    tag_data=str(ex),
                    tag_color="cyan",
                    to_reset_data=True,
                    to_new_line_data=True,
                )
                raise ex

        one_liner.one_line(
            tag="Training Complete",
            tag_color="cyan",
            to_reset_data=True,
            to_new_line_data=True,
        )
        self.callbacks.on_end()

    def state_train(
        self, step: int, progress_bar
    ) -> Tuple[Union[int, float], dict, int, tqdm.std.tqdm]:

        report_each = 100
        batch_loss = []
        mean_loss = 0

        for train_data in self.loader.train_loader:
            batch_logs = dict()
            self.callbacks.on_batch_begin(step, logs=batch_logs)

            train_data = gpu_variable(train_data)

            prediction = self.model(train_data["images"])
            calculated_loss = self.criterion(train_data["ground_truth"], prediction)
            self.optimizer.zero_grad()
            calculated_loss.backward()
            self.optimizer.step()

            batch_loss.append(calculated_loss.item())
            mean_loss = np.mean(batch_loss[-report_each:])
            batch_logs = handle_dictionary(
                batch_logs, "plt_lr", {"data": mean_loss, "tag": "Loss/Step"}
            )
            batch_logs = handle_dictionary(batch_logs, "model", self.model)
            batch_logs = handle_dictionary(
                batch_logs, "test_loader", self.loader.test_loader
            )
            self.callbacks.on_batch_end(step, logs=batch_logs)
            progress_bar.update(self.loader.train_loader.batch_size)
            progress_bar.set_postfix(loss="{:.5f}".format(mean_loss))
            step += 1
            self.metrics.get_metrics(
                ground_truth=train_data["ground_truth"], prediction=prediction
            )

        return mean_loss.item(), self.metrics.compute_mean(), step, progress_bar

    @torch.no_grad()
    def state_validate(self) -> Tuple[np.ndarray, dict]:
        logger.debug("Validation In Progress")
        losses = []
        start = time.time()
        for ongoing_count, val_data in enumerate(self.loader.val_loader):
            ongoing_count += 1

            one_liner.one_line(
                tag="Validation",
                tag_data=f"{ongoing_count}/{len(self.loader.val_loader)} "
                f"ETA -- {compute_eta(start, ongoing_count, len(self.loader.val_loader))}",
                tag_color="cyan",
                to_reset_data=True,
            )
            val_data = gpu_variable(val_data)

            prediction = self.model(val_data["images"])
            loss = self.criterion(val_data["ground_truth"], prediction)

            losses.append(loss.item())
            self.metrics.get_metrics(
                ground_truth=val_data["ground_truth"], prediction=prediction
            )
            

        return np.mean(losses), self.metrics.compute_mean()

    def collect_state(
        self,
        ongoing_epoch: int,
        end_epoch: int,
        step: int,
        bst_vld_loss: float,
        run_state: str,
    ) -> dict:
        assert run_state in ["interruption", "complete"], (
            "Expected state to save ['interruption', 'complete']" "got %s",
            (run_state,),
        )

        state_data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
            if self.optimizer is not None
            else "NA",
            "start_epoch": ongoing_epoch + 1
            if run_state == "complete"
            else ongoing_epoch,
            "step": step,
            "bst_vld_loss": bst_vld_loss if bst_vld_loss is not None else "NA",
            "end_epoch": end_epoch,
        }
        return {"state": state_data}


class EnhancedTrainingVisualizer:
    def __init__(self, save_dir='./training_plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_precision': [],
            'val_precision': [],
            'train_f1': [],
            'val_f1': [],
            'train_recall': [],
            'val_recall': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rate': []
        }
        
        # For AUC calculations
        self.all_predictions = []
        self.all_targets = []
    
    def update_history(self, epoch, epoch_logs):
        """Update training history with current epoch results"""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(epoch_logs['train_loss'])
        self.history['val_loss'].append(epoch_logs['valid_loss'])
        self.history['learning_rate'].append(epoch_logs['lr'])
        
        # Extract metrics
        train_metrics = epoch_logs['train_metric']
        val_metrics = epoch_logs['valid_metric']
        
        for metric in ['accuracy', 'precision', 'f1', 'recall', 'iou']:
            self.history[f'train_{metric}'].append(train_metrics[metric])
            self.history[f'val_{metric}'].append(val_metrics[metric])
    
    def collect_predictions(self, predictions, targets):
        """Collect predictions and targets for AUC calculation"""
        # Convert to numpy and flatten
        if torch.is_tensor(predictions):
            predictions = predictions.sigmoid().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        self.all_predictions.extend(predictions.flatten())
        self.all_targets.extend(targets.flatten())
    
    def reset_auc_data(self):
        """Reset AUC data collection"""
        self.all_predictions = []
        self.all_targets = []
    
    def plot_training_curves(self, save_name='training_curves.png'):
        """Plot comprehensive training curves"""
        fig = plt.figure(figsize=(20, 12))
        
        epochs = self.history['epochs']
        
        # Create 3x3 subplot grid
        # Row 1: Loss, Accuracy, Learning Rate
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(epochs, self.history['train_accuracy'], 'b-', label='Train', linewidth=2)
        ax2.plot(epochs, self.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Row 2: Precision, Recall, F1
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(epochs, self.history['train_precision'], 'b-', label='Train', linewidth=2)
        ax4.plot(epochs, self.history['val_precision'], 'r-', label='Validation', linewidth=2)
        ax4.set_title('Precision', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Precision')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(epochs, self.history['train_recall'], 'b-', label='Train', linewidth=2)
        ax5.plot(epochs, self.history['val_recall'], 'r-', label='Validation', linewidth=2)
        ax5.set_title('Recall', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Recall')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(epochs, self.history['train_f1'], 'b-', label='Train', linewidth=2)
        ax6.plot(epochs, self.history['val_f1'], 'r-', label='Validation', linewidth=2)
        ax6.set_title('F1 Score', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('F1 Score')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Row 3: IoU and summary stats
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(epochs, self.history['train_iou'], 'b-', label='Train', linewidth=2)
        ax7.plot(epochs, self.history['val_iou'], 'r-', label='Validation', linewidth=2)
        ax7.set_title('IoU (Intersection over Union)', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('IoU')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Summary statistics
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        summary_text = f"""
        Training Summary
        ─────────────────
        Total Epochs: {len(epochs)}
        
        Best Validation Loss: {min(self.history['val_loss']):.4f}
        @ Epoch: {self.history['val_loss'].index(min(self.history['val_loss'])) + 1}
        
        Best Validation IoU: {max(self.history['val_iou']):.4f}
        @ Epoch: {self.history['val_iou'].index(max(self.history['val_iou'])) + 1}
        
        Final Train Loss: {self.history['train_loss'][-1]:.4f}
        Final Val Loss: {self.history['val_loss'][-1]:.4f}
        
        Final Train IoU: {self.history['train_iou'][-1]:.4f}
        Final Val IoU: {self.history['val_iou'][-1]:.4f}
        """
        ax8.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        # Loss comparison bar chart
        ax9 = plt.subplot(3, 3, 9)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'IoU']
        train_vals = [self.history['train_accuracy'][-1],
                     self.history['train_precision'][-1],
                     self.history['train_recall'][-1],
                     self.history['train_f1'][-1],
                     self.history['train_iou'][-1]]
        val_vals = [self.history['val_accuracy'][-1],
                   self.history['val_precision'][-1],
                   self.history['val_recall'][-1],
                   self.history['val_f1'][-1],
                   self.history['val_iou'][-1]]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax9.bar(x - width/2, train_vals, width, label='Train', color='blue', alpha=0.7)
        ax9.bar(x + width/2, val_vals, width, label='Validation', color='red', alpha=0.7)
        ax9.set_ylabel('Score')
        ax9.set_title('Final Epoch Metrics', fontsize=14, fontweight='bold')
        ax9.set_xticks(x)
        ax9.set_xticklabels(metrics, rotation=45, ha='right')
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Training Progress Dashboard', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
        plt.close()
    
    def plot_auc_curves(self, save_name='auc_curves.png'):
        """Plot ROC and Precision-Recall curves"""
        if len(self.all_predictions) == 0:
            print("No predictions collected for AUC calculation")
            return None, None
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets).astype(int)
        
        # Compute ROC curve
        fpr, tpr, roc_thresholds = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Compute PR curve
        precision, recall, pr_thresholds = precision_recall_curve(targets, predictions)
        pr_auc = auc(recall, precision)
        
        # Find optimal threshold (Youden's J statistic)
        J = tpr - fpr
        optimal_idx = np.argmax(J)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2.5, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        baseline = targets.mean()
        ax2.plot(recall, precision, color='darkorange', lw=2.5,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        ax2.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                   label=f'Baseline (AP = {baseline:.4f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Area Under the Curve Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"AUC curves saved to: {save_path}")
        print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        plt.close()
        
        return roc_auc, pr_auc, optimal_threshold
    
