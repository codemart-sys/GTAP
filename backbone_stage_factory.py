from openpoints.models.build import MODELS
from openpoints.models.backbone.pointnext import PointNextEncoder

@MODELS.register_module()
class FourStageFixedInterfaceEncoder(PointNextEncoder):

    def __init__(self, backbone_channels=(24, 32, 64, 384), **kwargs):
        channels = tuple(backbone_channels)
        if channels not in ((24, 32, 64, 384), (24, 48, 96, 384)):
            raise ValueError('Undeclared original-interface channel layout')
        if kwargs.get('width') != 24 or tuple(kwargs.get('blocks', ())) != (1, 1, 1, 1) or tuple(kwargs.get('strides', ())) != (1, 4, 4, 4):
            raise ValueError('Only remove one encoder stage; retain remaining stage sampling parameters')
        self._declared_channels = channels
        self._stage_index = 0
        super().__init__(**kwargs)
        self.channel_list = list(channels)
        self.out_channels = 384
        assert self._stage_index == 4

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        target = self._declared_channels[self._stage_index]
        self._stage_index += 1
        return super()._make_enc(block, target, blocks, stride, group_args, is_head)
