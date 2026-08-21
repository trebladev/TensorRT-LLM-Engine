# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tempfile
import unittest

# isort: off
import tensorrt_llm
import tensorrt as trt
from tensorrt_llm.builder import BuildConfig, KVCacheType
from tensorrt_llm.functional import PositionEmbeddingType
# isort: on


class MyAddModule(tensorrt_llm.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class TestBuilder(unittest.TestCase):

    def test_mrope_position_embedding_enables_mrope_runtime_buffers(self):
        build_config = BuildConfig()

        build_config.update_use_mrope(PositionEmbeddingType.mrope)

        self.assertTrue(build_config.use_mrope)

    def test_attention_linear_hybrid_uses_paged_kv_and_state(self):
        build_config = BuildConfig()

        build_config.update_kv_cache_type('Qwen35ForCausalLM',
                                          ['linear', 'linear', 'attention'])

        self.assertEqual(build_config.kv_cache_type, KVCacheType.PAGED)
        self.assertTrue(build_config.plugin_config.paged_kv_cache)
        self.assertTrue(build_config.plugin_config.paged_state)

    def test_update_kv_cache_type_without_layer_types(self):
        build_config = BuildConfig()

        build_config.update_kv_cache_type('LlamaForCausalLM')

        self.assertEqual(build_config.kv_cache_type, KVCacheType.PAGED)
        self.assertTrue(build_config.plugin_config.paged_kv_cache)
        self.assertFalse(build_config.plugin_config.paged_state)

    def test_basic_builder_flow(self):
        tensorrt_llm.logger.set_level('verbose')
        builder = tensorrt_llm.Builder()
        builder_config = builder.create_builder_config("test", "llmTimingCache")
        builder_config.trt_builder_config.set_flag(trt.BuilderFlag.REFIT)
        model = MyAddModule()

        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            assert tensorrt_llm.default_net() == network
            assert tensorrt_llm.default_trtnet() == network.trt_network
            x = tensorrt_llm.Tensor(name='x', dtype=trt.float32, shape=[1, 1])
            y = tensorrt_llm.Tensor(name='y', dtype=trt.float32, shape=[1, 1])
            # Prepare
            network.set_named_parameters(model.named_parameters())
            # Forward
            z = model(x, y)
            z.mark_output('z', trt.float32)

        with self.assertRaises(AssertionError):
            tensorrt_llm.default_net()
        engine = builder.build_engine(network, builder_config)
        assert engine is not None
        refit_engine = builder.refit_engine(network, engine)
        assert refit_engine is not None
        builder.save_config(builder_config, tempfile.mktemp())

    def test_top_level_dont_have_functional_apis(self):
        # This did not check all the functional apis, but should already prevent
        # from .functional import * in the __init__.py
        with self.assertRaises(AttributeError):
            x = tensorrt_llm.activation
            x = tensorrt_llm.assertion
            x = tensorrt_llm.einsum
            print(x)  # to avoid the delete of x
        x = tensorrt_llm.functional.activation
        x = tensorrt_llm.functional.assertion
        x = tensorrt_llm.functional.einsum
        print(x)  # to avoid the delete of x


class TestSubprocess(unittest.TestCase):

    def import_using_popen(self):
        import tensorrt_llm  # isort: skip
        from subprocess import Popen

        Popen(["python3", "-c", "import tensorrt_llm"])


if __name__ == '__main__':
    unittest.main()
