# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings

from mmcv.runner import DefaultOptimizerConstructor, get_dist_info
from mmpose.utils import get_root_logger
from mmcv.runner import OPTIMIZER_BUILDERS
        
        
def get_num_layer_for_swin(var_name, num_max_layer, layers_per_stage, layers_from_stage):
    if var_name in ("backbone.cls_token", "backbone.mask_token",
                    "backbone.pos_embed", "backbone.absolute_pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.layers"):
        if var_name.split('.')[3] == "blocks":
            if layers_from_stage:
                stage = int(layers_from_stage.split('.')[1])
                if int(var_name.split('.')[2]) > stage:
                    return num_max_layer - 1
                elif int(var_name.split('.')[2]) == stage:
                    block = int(layers_from_stage.split('.')[3])
                    if int(var_name.split('.')[4]) >= block:
                        return num_max_layer - 1
            stage_id = int(var_name.split('.')[2])
            layer_id = int(var_name.split('.')[4]) \
                    + sum(layers_per_stage[:stage_id])
            return layer_id + 1
        elif var_name.split('.')[3] == "downsample":
            if layers_from_stage:
                stage = int(layers_from_stage.split('.')[1])
                if int(var_name.split('.')[2]) >= stage:
                    return num_max_layer - 1
            stage_id = int(var_name.split('.')[2])
            layer_id = sum(layers_per_stage[:stage_id + 1])
            return layer_id
    else:
        return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module()
class SwinLayerDecayOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        print(self.paramwise_cfg)
        layers_from_stage = self.paramwise_cfg.get('from_stage', None)
        layers_per_stage = self.paramwise_cfg.get('num_layers')
        no_decay_names = self.paramwise_cfg.get('no_decay_names', [])
        for i in range(len(layers_per_stage) - 1):
            layers_per_stage[i] = layers_per_stage[i] + 1  # patch merging
        num_layers = sum(layers_per_stage) + 2  # 2: patch embed, head
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        print("Build SwinLayerDecayOptimizerConstructor %f - %d" % (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in ('absolute_pos_embed'):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

                for nd_name in no_decay_names:
                    if nd_name in name:
                        group_name = "no_decay"
                        this_weight_decay = 0.
                        break

            layer_id = get_num_layer_for_swin(name, num_layers, layers_per_stage, layers_from_stage)
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * self.base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"],
                    "lr_scale": parameter_groups[key]["lr_scale"],
                    "lr": parameter_groups[key]["lr"],
                    "weight_decay": parameter_groups[key]["weight_decay"],
                }
            print("Param groups = %s" % json.dumps(to_display, indent=2))

        # state_dict = module.state_dict()
        # for group_name in parameter_groups:
        #     group = parameter_groups[group_name]
        #     for name in group["param_names"]:
        #         group["params"].append(state_dict[name])
        params.extend(parameter_groups.values())