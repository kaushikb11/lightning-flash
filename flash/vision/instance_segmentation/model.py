# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

import torch
import torchvision
from torch import nn
from torch.optim import Optimizer
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_iou

from flash.core import Task
from flash.core.data import DataPipeline
from flash.vision.detection.data import ImageDetectorDataPipeline
from flash.vision.detection.finetuning import ImageDetectorFineTuning

_models = {"maskrcnn_resnet50_fpn": torchvision.models.detection.maskrcnn_resnet50_fpn}


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


class ImageInstanceSegmnetation(Task):
    """Image Instance Segmnetaion task
    Args:
        num_classes: the number of classes for instance segmentation, including background
        model: either a string of :attr`_models` or a custom nn.Module.
            Defaults to 'maskrcnn_resnet50_fpn'.
        loss: the function(s) to update the model with. Has no effect for torchvision detection models.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger.
            Defaults to None.
        optimizer: The optimizer to use for training. Can either be the actual class or the class name.
            Defaults to Adam.
        pretrained: Whether the model from torchvision should be loaded with it's pretrained weights.
            Has no effect for custom models. Defaults to True.
        learning_rate: The learning rate to use for training
    """

    def __init__(
        self,
        num_classes: int,
        model: Union[str, nn.Module] = "maskrcnn_resnet50_fpn",
        loss=None,
        metrics: Union[Callable, nn.Module, Mapping, Sequence, None] = None,
        optimizer: Type[Optimizer] = torch.optim.Adam,
        pretrained: bool = True,
        learning_rate=1e-3,
        **kwargs,
    ):

        self.save_hyperparameters()

        if model in _models:
            model = _models[model](pretrained=pretrained)
            if isinstance(model, torchvision.models.detection.MaskRCNN):
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                head = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
                model.roi_heads.box_predictor = head

                in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        super().__init__(
            model=model,
            loss_fn=loss,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

    def training_step(self, batch, batch_idx) -> Any:
        """The training step.
        Overrides Task.training_step
        """
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        return {"avg_val_iou": avg_iou, "log": logs}

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"test_iou": iou}

    def test_epoch_end(self, outs):
        avg_iou = torch.stack([o["test_iou"] for o in outs]).mean()
        logs = {"test_iou": avg_iou}
        return {"avg_test_iou": avg_iou, "log": logs}
