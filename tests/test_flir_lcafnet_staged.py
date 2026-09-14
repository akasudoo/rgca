import unittest

import torch.nn as nn

from train import set_flir_lcafnet_migration_stage


class _ToyC0RGCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.yaml = {'architecture_variant': 'c0_full_dual_rgca'}
        self.model = nn.ModuleList([nn.Linear(1, 1) for _ in range(46)])


class FLIRLCAFNetStagedTrainingTest(unittest.TestCase):
    def test_fusion_only_then_fusion_head_scope(self):
        model = _ToyC0RGCA()

        set_flir_lcafnet_migration_stage(model, fusion_only=True)
        for index, module in enumerate(model.model):
            expected = 20 <= index <= 23
            self.assertTrue(all(p.requires_grad == expected
                                for p in module.parameters()))

        set_flir_lcafnet_migration_stage(model, fusion_only=False)
        for index, module in enumerate(model.model):
            expected = 20 <= index <= 45
            self.assertTrue(all(p.requires_grad == expected
                                for p in module.parameters()))


if __name__ == '__main__':
    unittest.main()
