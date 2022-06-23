import torch
import pytorch_lightning as pl

from itertools import chain
from src.models.GATsSPG_architectures.GATs_SuperGlue import GATsSuperGlue
from src.losses.focal_loss import FocalLoss
from src.utils.eval_utils import compute_query_pose_errors, aggregate_metrics
from src.utils.vis_utils import draw_reprojection_pair
from src.utils.comm import gather
from src.models.extractors.SuperPoint.superpoint import SuperPoint
from src.sfm.extract_features import confs
from src.utils.model_io import load_network


class LitModelGATsSPG(pl.LightningModule):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        self.extractor = SuperPoint(confs['superpoint'])
        load_network(self.extractor, self.hparams.spp_model_path, force=False)
        self.matcher = GATsSuperGlue(hparams=self.hparams)
        self.crit = FocalLoss(
            alpha=self.hparams.focal_loss_alpha, 
            gamma=self.hparams.focal_loss_gamma, 
            neg_weights=self.hparams.neg_weights, 
            pos_weights=self.hparams.pos_weights
        )
        self.n_vals_plot = 10

        self.train_loss_hist = []
        self.val_loss_hist = []
        self.save_flag = True
    
    def forward(self, x):
        return self.matcher(x)

    def training_step(self, batch, batch_idx):
        self.save_flag = False
        data, conf_matrix_gt = batch
        preds, conf_matrix_pred = self.matcher(data)

        loss_mean = self.crit(conf_matrix_pred, conf_matrix_gt)
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            self.logger.experiment.add_scalar('train/loss', loss_mean, self.global_step)
        
        return {'loss': loss_mean, 'preds': preds}
    
    def validation_step(self, batch, batch_idx):
        data, _ = batch
        extraction = self.extractor(data['image'])
        data.update({
            'keypoints2d': extraction['keypoints'][0][None],
            'descriptors2d_query': extraction['descriptors'][0][None],
        })
        preds, conf_matrix_pred = self.matcher(data)

        pose_pred, val_results = compute_query_pose_errors(data, preds)

        # Visualize match:
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {'evaluation': []}
        if batch_idx % val_plot_interval == 0:
            figures = draw_reprojection_pair(data, val_results, visual_color_type='conf')

        loss_mean = 0
        self.log('val/loss', loss_mean, on_step=False, on_epoch=True, prog_bar=False)
        del data
        return {'figures': figures, 'metrics': val_results}
    
    def test_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss, global_step=self.current_epoch
            )
    
    def validation_epoch_end(self, outputs):
        self.val_loss_hist.append(self.trainer.callback_metrics['val/loss'])
        self.log('val/loss_best', min(self.val_loss_hist), prog_bar=False)

        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        for valset_idx, outputs in enumerate(multi_outputs):
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.sanity_checking:
                cur_epoch = -1
            
            def flattenList(x): return list(chain(*x))

            # Log val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

            # Log figures:
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # NOTE: tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                val_metrics_4tb = aggregate_metrics(metrics)
                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f'metrics_{valset_idx}/{k}', v, global_step=cur_epoch)
                
                for k, v in figures.items():
                    for plot_idx, fig in enumerate(v):
                        self.logger.experiment.add_figure(
                            f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True
                        )
    
    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=self.hparams.milestones,
                                                                gamma=self.hparams.gamma)
            return [optimizer], [lr_scheduler]
        else:
            raise Exception("Invalid optimizer name.")