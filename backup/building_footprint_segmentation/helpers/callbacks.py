import datetime
import logging
import os
import shutil

import cv2
import torch
import warnings
import time

from py_oneliner import one_liner

from building_footprint_segmentation.utils import date_time
from building_footprint_segmentation.utils.operations import (
    is_overridden_func,
    make_directory,
)
from building_footprint_segmentation.utils.py_network import (
    adjust_model,
    gpu_variable,
    convert_tensor_to_numpy,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("segmentation")


class CallbackList(object):
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        if len(callbacks) != 0:
            [
                logger.debug("Registered {}".format(c.__class__.__name__))
                for c in callbacks
            ]

    def append(self, callback):
        logger.debug("Registered {}".format(callback.__class__.__name__))
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Epoch Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_epoch_begin):
                logger.debug(
                    "Nothing Registered On Epoch Begin {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Epoch End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_epoch_end):
                logger.debug(
                    "Nothing Registered On Epoch End {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            logger.debug("On Batch Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_batch_begin):
                logger.debug(
                    "Nothing Registered On Batch Begin {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):

        for callback in self.callbacks:
            logger.debug("On Batch End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_batch_end):
                logger.debug(
                    "Nothing Registered On Batch End {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.on_batch_end(batch, logs)

    def on_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On Begin {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_begin):
                logger.debug(
                    "Nothing Registered On Begin {}".format(callback.__class__.__name__)
                )
            callback.on_begin(logs)

    def on_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("On End {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.on_end):
                logger.debug(
                    "Nothing Registered On End {}".format(callback.__class__.__name__)
                )
            callback.on_end(logs)

    def interruption(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            logger.debug("Interruption {}".format(callback.__class__.__name__))
            if not is_overridden_func(callback.interruption):
                logger.debug(
                    "Nothing Registered On Interruption {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.interruption(logs)

    def update_params(self, params):
        for callback in self.callbacks:
            if not is_overridden_func(callback.update_params):
                logger.debug(
                    "Nothing Registered On Update param {}".format(
                        callback.__class__.__name__
                    )
                )
            callback.update_params(params)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    def __init__(self, log_dir):
        self.log_dir = os.path.join(
            log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_begin(self, logs=None):
        pass

    def on_end(self, logs=None):
        pass

    def interruption(self, logs=None):
        pass

    def update_params(self, params):
        pass


class TrainStateCallback(Callback):
    """
    Save the training state
    """

    def __init__(self, log_dir):
        super().__init__(log_dir)
        state = make_directory(self.log_dir, "state")
        self.chk = os.path.join(state, "default.pt")
        self.best = os.path.join(state, "best.pt")

        self.previous_best = None

    def on_epoch_end(self, epoch, logs=None):
        valid_loss = logs["valid_loss"]
        my_state = logs["state"]
        if self.previous_best is None or valid_loss < self.previous_best:
            self.previous_best = valid_loss
            torch.save(my_state, str(self.best))
        torch.save(my_state, str(self.chk))
        logger.debug(
            "Successful on Epoch End {}, Saved State".format(self.__class__.__name__)
        )

    def interruption(self, logs=None):
        my_state = logs["state"]

        torch.save(my_state, str(self.chk))
        logger.debug(
            "Successful on Interruption {}, Saved State".format(self.__class__.__name__)
        )


class TensorBoardCallback(Callback):
    """
    Log tensor board events
    """

    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.writer = SummaryWriter(make_directory(self.log_dir, "events"))

    def plt_scalar(self, y, x, tag):
        if type(y) is dict:
            self.writer.add_scalars(tag, y, global_step=x)
            self.writer.flush()
        else:
            self.writer.add_scalar(tag, y, global_step=x)
            self.writer.flush()

    def plt_images(self, img, global_step, tag):
        self.writer.add_image(tag, img, global_step)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        lr = logs["lr"]
        train_loss = logs["train_loss"]
        valid_loss = logs["valid_loss"]

        train_metric = logs["train_metric"]
        valid_metric = logs["valid_metric"]

        self.plt_scalar(lr, epoch, "LR/Epoch")
        self.plt_scalar(
            {"train_loss": train_loss, "valid_loss": valid_loss}, epoch, "Loss/Epoch"
        )

        metric_keys = list(train_metric.keys())
        for key in metric_keys:
            self.plt_scalar(
                {
                    "Train_{}".format(key): train_metric[key],
                    "Valid_{}".format(key): valid_metric[key],
                },
                epoch,
                "{}/Epoch".format(key),
            )

        logger.debug(
            "Successful on Epoch End {}, Data Plot".format(self.__class__.__name__)
        )

    def on_batch_end(self, batch, logs=None):
        img_data = logs["plt_img"] if "plt_img" in logs else None
        data = logs["plt_lr"]

        if img_data is not None:
            # self.plt_images(to_tensor(np.moveaxis(img_data["img"], -1, 0)), batch, img_data["tag"])
            pass

        self.plt_scalar(data["data"], batch, data["tag"])
        logger.debug(
            "Successful on Batch End {}, Data Plot".format(self.__class__.__name__)
        )


class SchedulerCallback(Callback):
    def __init__(self, scheduler):
        super().__init__(None)
        self.scheduler = scheduler

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step(epoch)
        logger.debug(
            "Successful on Epoch End {}, Lr Scheduled".format(self.__class__.__name__)
        )


class TimeCallback(Callback):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.start_time = None

    def on_begin(self, logs=None):
        self.start_time = time.time()

    def on_end(self, logs=None):
        end_time = time.time()
        total_time = date_time.get_time(end_time - self.start_time)
        one_liner.one_line(
            tag="Run Time",
            tag_data=f"{total_time}",
            tag_color="cyan",
            to_reset_data=True,
            to_new_line_data=True,
        )

    def interruption(self, logs=None):
        end_time = time.time()
        total_time = date_time.get_time(end_time - self.start_time)
        one_liner.one_line(
            tag="Run Time",
            tag_data=f"{total_time}",
            tag_color="cyan",
            to_reset_data=True,
            to_new_line_data=True,
        )


class TrainChkCallback(Callback):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.chk = os.path.join(make_directory(self.log_dir, "chk_pth"), "chk_pth.pt")

    def on_epoch_end(self, epoch, logs=None):
        my_state = logs["state"]
        #torch.save(my_state["model"], str(self.chk))
        torch.save(adjust_model(my_state["model"]), str(self.chk))
        logger.debug(
            "Successful on Epoch End {}, Chk Saved".format(self.__class__.__name__)
        )

    def interruption(self, logs=None):
        my_state = logs["state"]
        #torch.save(my_state["model"], str(self.chk))
        torch.save(adjust_model(my_state["model"]), str(self.chk))
        logger.debug(
            "Successful on interruption {}, Chk Saved".format(self.__class__.__name__)
        )
#torch.save(adjust_model(my_state["model"]), str(self.chk)) changed this for possible model corruption

class TestDuringTrainingCallback(Callback):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.test_path = os.path.join(self.log_dir, "test_on_epoch_end")

    def on_epoch_end(self, epoch, logs=None):
        model = logs["model"]
        test_loader = logs["test_loader"]
        model.eval()
        try:
            if os.path.exists(self.test_path):
                shutil.rmtree(self.test_path)

            for i, test_data in enumerate(test_loader):
                self.inference(
                    model,
                    gpu_variable(test_data["images"]),
                    test_data["file_name"],
                    make_directory(
                        os.path.dirname(self.test_path),
                        os.path.basename(self.test_path),
                    ),
                    epoch,
                )
                break
        except Exception as ex:
            logger.exception("Skipped Exception in {}".format(self.__class__.__name__))
            logger.exception("Exception {}".format(ex))
            pass

    def inference(self, model, image, file_name, save_path, index):
        pass


class BinaryTestCallback(TestDuringTrainingCallback):
    def __init__(self, log_dir, threshold: float = 0.20):
        super().__init__(log_dir)
        self._threshold = threshold

    @torch.no_grad()
    def inference(self, model, image, file_name, save_path, index):
        """

        :param model: the model used for training
        :param image: the images loaded by the test loader
        :param file_name: the file name of the test image
        :param save_path: path where to save the image
        :param index:
        :return:
        """
        prediction = model(image)
        prediction = prediction.sigmoid()
        prediction[prediction >= self._threshold] = 1
        prediction[prediction < self._threshold] = 0

        batch, _, h, w = prediction.shape
        for i in range(batch):
            prediction_numpy = convert_tensor_to_numpy(prediction[i])
            prediction_numpy = prediction_numpy.reshape((h, w))
            cv2.imwrite(
                os.path.join(save_path, f"{file_name[i]}.png"), prediction_numpy * 255
            )


def load_default_callbacks(log_dir: str):
    return [
        TrainChkCallback(log_dir),
        TimeCallback(log_dir),
        TensorBoardCallback(log_dir),
        TrainStateCallback(log_dir),
    ]


def load_callback(log_dir: str, callback: str) -> Callback:
    """
    :param log_dir:
    :param callback:
    :return:
    """
    return eval(callback)(log_dir)



class VisualizationCallback(Callback):
    """Callback to visualize training progress and generate AUC curves"""
    
    def __init__(self, log_dir, visualizer, plot_frequency=5, collect_auc_data=True):
        """
        Args:
            log_dir: Directory to save logs
            visualizer: EnhancedTrainingVisualizer instance
            plot_frequency: Plot curves every N epochs (default: 5)
            collect_auc_data: Whether to collect predictions for AUC calculation
        """
        super().__init__(log_dir)
        self.visualizer = visualizer
        self.plot_frequency = plot_frequency
        self.collect_auc_data = collect_auc_data
        self.current_epoch_predictions = []
        self.current_epoch_targets = []
    
    def on_epoch_begin(self, epoch, logs=None):
        """Reset AUC data collection at the start of each epoch"""
        if self.collect_auc_data:
            self.current_epoch_predictions = []
            self.current_epoch_targets = []
    
    def on_batch_end(self, batch, logs=None):
        """
        Collect predictions during validation (optional for AUC calculation)
        Note: This would need to be called during validation, not training
        """
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        """Update visualization at the end of each epoch"""
        if logs:
            # Update training history
            self.visualizer.update_history(epoch, logs)
            
            # Plot curves periodically
            if (epoch + 1) % self.plot_frequency == 0:
                print(f"\nGenerating training curves at epoch {epoch + 1}...")
                self.visualizer.plot_training_curves(
                    save_name=f'training_curves_epoch_{epoch+1}.png'
                )
    
    def on_end(self, logs=None):
        """Generate final plots when training completes"""
        print("\n" + "="*60)
        print("Training Complete - Generating Final Visualizations")
        print("="*60)
        
        # Generate final training curves
        self.visualizer.plot_training_curves(save_name='final_training_curves.png')
        
        # Save metrics to CSV
        #self.visualizer.save_metrics_csv('training_metrics.csv')
        
        print("Visualization complete!")
    
    def interruption(self, logs=None):
        """Generate plots when training is interrupted"""
        print("\n" + "="*60)
        print("Training Interrupted - Saving Current Visualizations")
        print("="*60)
        
        if len(self.visualizer.history['epochs']) > 0:
            self.visualizer.plot_training_curves(save_name='interrupted_training_curves.png')
            self.visualizer.save_metrics_csv('interrupted_metrics.csv')


class AUCCallback(Callback):
    """Separate callback to compute AUC during validation"""
    
    def __init__(self, log_dir, visualizer, compute_frequency=10):
        """
        Args:
            log_dir: Directory to save logs
            visualizer: EnhancedTrainingVisualizer instance
            compute_frequency: Compute AUC every N epochs (default: 10)
        """
        super().__init__(log_dir)
        self.visualizer = visualizer
        self.compute_frequency = compute_frequency
    
    def update_params(self, params):
        """Store reference to the loader object"""
        if 'loader' in params:
            self.trainer_loader = params['loader']

    def on_epoch_end(self, epoch, logs=None):
        """Compute and plot AUC curves periodically"""
        if (epoch + 1) % self.compute_frequency == 0:
            if logs and 'model' in logs :
                print(f"\nComputing AUC curves at epoch {epoch + 1}...")
                
                model = logs['model']
                val_loader = None
                if self.trainer_loader is not None:
                    val_loader = self.trainer_loader.val_loader
                elif 'val_loader' in logs:
                    val_loader = logs['val_loader']
                
                if val_loader is not None:
                    # Collect predictions on validation set
                    self.visualizer.reset_auc_data()
                    self._collect_predictions(model, val_loader)                
                
                # Collect predictions on test set
                #self.visualizer.reset_auc_data()
                #self._collect_predictions(model, test_loader)
                
                # Plot AUC curves
                roc_auc, pr_auc, optimal_thresh = self.visualizer.plot_auc_curves(
                    save_name=f'auc_curves_epoch_{epoch+1}.png'
                )
    
    @torch.no_grad()
    def _collect_predictions(self, model, data_loader):
        """Collect predictions from the model"""
        model.eval()
        device = next(model.parameters()).device

        for batch in data_loader:
            if isinstance(batch, dict):
                images = batch['images'].to(device)
                targets = batch['ground_truth'].to(device)
            else:
                images, targets = batch
                images, targets = images.to(device), targets.to(device)
            
            # Get predictions
            outputs = model(images)
            
            # Collect for AUC calculation
            self.visualizer.collect_predictions(outputs, targets)
    
    def on_end(self, logs=None):
        """Compute final AUC curves when training completes"""
        if logs and 'model' in logs and 'test_loader' in logs:
            print("\nComputing final AUC curves...")
            
            model = logs['model']
            test_loader = logs['test_loader']
            
            # Collect predictions on test set
            self.visualizer.reset_auc_data()
            self._collect_predictions(model, test_loader)
            
            # Plot final AUC curves
            self.visualizer.plot_auc_curves(save_name='final_auc_curves.png')
