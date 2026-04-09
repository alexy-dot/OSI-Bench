"""
BAGEL Model Wrapper for VLMEvalKit
Adapts Omni-View's BAGEL model to VLMEvalKit's interface

Author: Adapted for Omni-View evaluation on OSI-Bench
"""

import os
import sys
import torch
from PIL import Image
from typing import List, Dict, Any, Optional

from vlmeval.vlm.base import BaseVLM


class BAGEL(BaseVLM):
    """
    BAGEL model wrapper for VLMEvalKit evaluation.
    
    This class wraps the Omni-View BAGEL model to work with VLMEvalKit's
    evaluation framework. It implements the generate_inner() method that
    VLMEvalKit calls during evaluation.
    """
    
    def __init__(
        self,
        model_path: str = './pretrained_model/BAGEL-7B-MoT/',
        safetensor_path: str = 'model.safetensors',
        max_length: int = 16,
        **kwargs
    ):
        """
        Initialize BAGEL model.
        
        Args:
            model_path: Path to BAGEL model directory
            safetensor_path: Path to safetensors file (relative to model_path or absolute)
            max_length: Maximum generation length
        """
        super().__init__(**kwargs)
        
        self.model_path = model_path
        self.max_length = max_length
        
        if not os.path.isabs(safetensor_path):
            safetensor_path = os.path.join(model_path, safetensor_path)
        
        self._load_model(model_path, safetensor_path)
    
    def _load_model(self, model_path: str, safetensor_path: str):
        """Load BAGEL model, tokenizer, and image transform."""
        
        import yaml
        from safetensors.torch import load_file
        
        from modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, 
            Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
        )
        from modeling.qwen2 import Qwen2Tokenizer
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        
        print(f"[BAGEL] Loading model from {model_path}")
        
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        
        config = BagelConfig(
            visual_gen=False,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
        )
        
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        self.model = Bagel(language_model, vit_model, config)
        self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
        
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)
        
        model_state_dict = load_file(safetensor_path, device="cpu")
        msg = self.model.load_state_dict(model_state_dict, strict=False)
        print(f"[BAGEL] Model loaded: {msg}")
        del model_state_dict
        
        self.model = self.model.cuda().eval()
        self.model = self.model.to(torch.bfloat16)
        
        with open("./data/configs/example.yaml", "r") as f:
            data_config = yaml.safe_load(f)
        
        max_image_size = data_config['vlm_sft']['image_transform_args']['max_image_size']
        min_image_size = data_config['vlm_sft']['image_transform_args']['min_image_size']
        image_stride = data_config['vlm_sft']['image_transform_args']['image_stride']
        max_pixels = data_config['vlm_sft']['image_transform_args']['max_pixels']
        
        self.image_transform = ImageTransform(
            max_image_size=max_image_size,
            min_image_size=min_image_size,
            image_stride=image_stride,
            max_pixels=max_pixels,
        )
        
        print(f"[BAGEL] Model initialization complete")
    
    def generate_inner(self, msgs: List[Dict], dataset: str = None) -> str:
        """
        Generate response from multimodal messages.
        
        Args:
            msgs: List of messages in VLMEvalKit format
                  Each message is a dict with 'type' ('image' or 'text') and 'value'
                  Example: [
                      {'type': 'image', 'value': '/path/to/image.jpg'},
                      {'type': 'image', 'value': '/path/to/image2.jpg'},
                      {'type': 'text', 'value': 'What is in the image?'}
                  ]
            dataset: Dataset name (optional)
        
        Returns:
            Model's text response
        """
        images = []
        prompt = ''
        
        for msg in msgs:
            if msg['type'] == 'image':
                img_value = msg['value']
                if isinstance(img_value, str):
                    if os.path.exists(img_value):
                        img = Image.open(img_value).convert('RGB')
                    else:
                        print(f"[BAGEL] Warning: Image path does not exist: {img_value}")
                        continue
                elif isinstance(img_value, Image.Image):
                    img = img_value.convert('RGB')
                else:
                    print(f"[BAGEL] Warning: Unknown image type: {type(img_value)}")
                    continue
                images.append(img)
            elif msg['type'] == 'text':
                prompt = msg['value']
        
        if not images:
            return ""
        
        with torch.no_grad():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
                image_transform=self.image_transform,
                images=images,
                prompt=prompt,
                max_length=self.max_length,
                do_sample=False,
                temperature=1.0,
            )
        
        return response
    
    def build_prompt(self, line: Any, dataset: str = None) -> List[Dict]:
        """
        Build prompt for the model.
        
        This method can be overridden to customize prompt format for specific datasets.
        """
        return super().build_prompt(line, dataset)


class BAGEL_7B_MoT(BAGEL):
    """BAGEL-7B-MoT model alias for easy configuration."""
    
    def __init__(self, **kwargs):
        if 'model_path' not in kwargs:
            kwargs['model_path'] = './pretrained_model/BAGEL-7B-MoT/'
        if 'safetensor_path' not in kwargs:
            kwargs['safetensor_path'] = 'model.safetensors'
        super().__init__(**kwargs)


class OmniView(BAGEL):
    """Omni-View model alias (same as BAGEL-7B-MoT)."""
    
    def __init__(self, **kwargs):
        if 'model_path' not in kwargs:
            kwargs['model_path'] = './pretrained_model/BAGEL-7B-MoT/'
        if 'safetensor_path' not in kwargs:
            kwargs['safetensor_path'] = 'model.safetensors'
        super().__init__(**kwargs)
