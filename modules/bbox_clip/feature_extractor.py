

__all__ =['FeatureExtractorBase','FeatureExtractor']


import torch
import copy





class LastLayerToExtractReachedException(Exception):
    pass
class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None



# class FeatureExtractorBase(object):
class FeatureExtractorBase(object):

    def __init__(self):
        super(FeatureExtractorBase, self).__init__()
        self.backbone=None
class FeatureExtractor(FeatureExtractorBase):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from):
        super(FeatureExtractor, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        # print(backbone)
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )

            pointers = extract_layer.split(".")

            def int_type(x):
                if x.isnumeric():
                    x = int(x)
                return x

            network_layer=backbone
            for index,p in enumerate(pointers):
                if p.isnumeric():
                    p = int(p)
                    network_layer=network_layer[p]
                else:
                    network_layer = network_layer.__dict__["_modules"][p]



            # if "." in extract_layer:
            #     extract_block, extract_idx = extract_layer.split(".")
            #     network_layer = backbone.__dict__["_modules"][extract_block]
            #     if extract_idx.isnumeric():
            #         extract_idx = int(extract_idx)
            #         network_layer = network_layer[extract_idx]
            #     else:
            #         network_layer = network_layer.__dict__["_modules"][extract_idx]
            # else:
            #     network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )

    def __call__(self, images,to_list=False):
        self.outputs.clear()
        # with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
        try:
            _ = self.backbone(images)
        except LastLayerToExtractReachedException:
            pass
        if not to_list:
            return self.outputs
        else:
            # print(self.outputs.keys())

            outs= [ self.outputs[key] for key in  self.layers_to_extract_from  ] 
            return outs

    # def feature_dimensions(self, input_shape):
    #     """Computes the feature dimensions for all layers given input_shape."""
    #     _input = torch.ones([1] + list(input_shape)).to(self.device)
    #     _output = self(_input)
    #     return [_output[layer].shape[1] for layer in self.layers_to_extract_from]
    def feature_dimensions(self, input_shape,device=None):
        """Computes the feature dimensions for all layers given input_shape."""
        # _input = torch.ones([1] + list(input_shape)).to(self.device)
        _input = torch.ones([1] + list(input_shape))
        if device is not None:
            _input=_input.to(device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]

# class FeatureExtractor(FeatureExtractorBase):
#     """Efficient extraction of network features."""

#     def __init__(self, backbone, layers_to_extract_from):
#         super(FeatureExtractorBase, self).__init__()
#         """Extraction of network features.

#         Runs a network only to the last layer of the list of layers where
#         network features should be extracted from.

#         Args:
#             backbone: torchvision.model
#             layers_to_extract_from: [list of str]
#         """
#         self.layers_to_extract_from = layers_to_extract_from
#         self.backbone = backbone
#         if not hasattr(backbone, "hook_handles"):
#             self.backbone.hook_handles = []
#         for handle in self.backbone.hook_handles:
#             handle.remove()
#         self.outputs = {}
#         # print(layers_to_extract_from)
#         for extract_layer in layers_to_extract_from:
#             forward_hook = ForwardHook(
#                 self.outputs, extract_layer, layers_to_extract_from[-1]
#             )
#             if "." in extract_layer:
#                 extract_block, extract_idx = extract_layer.split(".")
#                 network_layer = backbone.__dict__["_modules"][extract_block]
#                 if extract_idx.isnumeric():
#                     extract_idx = int(extract_idx)
#                     network_layer = network_layer[extract_idx]
#                 else:
#                     network_layer = network_layer.__dict__["_modules"][extract_idx]
#             else:
#                 network_layer = backbone.__dict__["_modules"][extract_layer]

#             if isinstance(network_layer, torch.nn.Sequential):
#                 self.backbone.hook_handles.append(
#                     network_layer[-1].register_forward_hook(forward_hook)
#                 )
#             else:
#                 self.backbone.hook_handles.append(
#                     network_layer.register_forward_hook(forward_hook)
#                 )
#         # self.to(self.device)

#     def feature_dimensions(self, input_shape,device=None):
#         """Computes the feature dimensions for all layers given input_shape."""
#         # _input = torch.ones([1] + list(input_shape)).to(self.device)
#         _input = torch.ones([1] + list(input_shape))
#         if device is not None:
#             _input=_input.to(device)
#         _output = self(_input)
#         return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


#     def forward(self, images,to_list=False):
#         self.outputs.clear()
#         with torch.no_grad():
#             # The backbone will throw an Exception once it reached the last
#             # layer to compute features from. Computation will stop there.
#             try:
#                 _ = self.backbone(images)
#             except LastLayerToExtractReachedException:
#                 pass
#         if not to_list:
#             return self.outputs
#         else:
#             # print(self.outputs.keys())
#             outs= [ self.outputs[key] for key in  self.layers_to_extract_from  ] 
#             return outs

