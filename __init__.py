 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from PIL import Image, ImageFilter
import math
import comfy.utils
import node_helpers
from einops import rearrange
import folder_paths
import os
from time import time

def tensor2image_sm(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image
def phi2narry_sm(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

class RefineAnything_Pasteback(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="RefineAnything_Pasteback",
            display_name="RefineAnything_Pasteback",
            category="RefineAnything",
            inputs=[
                io.Image.Input("generated_image" ),
                io.Conditioning.Input("cond" ),
                io.Int.Input("mask_grow",default=3, min=0, max=4097, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("blend_blur",default=5, min=0, max=4096, step=1, display_mode=io.NumberDisplay.number),
                io.Boolean.Input("adain",default=True),
                io.Boolean.Input("wavelet",default=False),
                io.Boolean.Input("save_rgba",default=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Image.Output(display_name="transparent_bg"),
            ]
            ,
            )
    @classmethod
    def execute(cls,generated_image,cond,mask_grow,blend_blur,adain,wavelet,save_rgba ) -> io.NodeOutput:
        def paste_back(
            original: Image.Image,
            generated: Image.Image,
            mask_l: Image.Image,
            crop_box: tuple[int, int, int, int] | None = None,
            mask_grow: int = 3,
            blend_blur: int = 5,
            wavelet: bool = False,
            adain: bool = False,
            save_rgba: bool = True,

            ) -> Image.Image:
            """Blend *generated* back into *original* through a smoothed mask."""
            m = mask_l.convert("L")
            if mask_grow > 0:
                m = m.filter(ImageFilter.MaxFilter(size=2 * mask_grow + 1))
            if blend_blur > 0:
                m = m.filter(ImageFilter.GaussianBlur(radius=float(blend_blur)))

            target = original.crop(crop_box) if crop_box else original
            dst = np.asarray(target.convert("RGB")).astype(np.float32)
            
            alpha = np.asarray(m.resize(target.size, Image.BILINEAR)).astype(np.float32) / 255.0
            # blended = src * alpha[:, :, None] + dst * (1.0 - alpha[:, :, None])
            if  adain:
                from .align_color import adain_color_fix
                src =adain_color_fix(generated.convert("RGB").resize(target.size, Image.BICUBIC) , target)
                src=np.asarray(src).astype(np.float32)
            else:
                src = np.asarray(
                generated.convert("RGB").resize(target.size, Image.BICUBIC)
            ).astype(np.float32)
            blended = src * alpha[:, :, None] + dst * (1.0 - alpha[:, :, None])
            composited = Image.fromarray(
                np.clip(blended, 0, 255).astype(np.uint8), mode="RGB"
            )
            transparent_bg = Image.new('RGBA', original.size, (0, 0, 0, 0))
            prefix = f"composited_{int(time())}"
            if wavelet:
                from .align_color import wavelet_reconstruction
                x1 =  phi2narry_sm(composited).permute(0, 3, 1, 2) #--> torch.Size([1, 3, 1024, 1024])
                x1 = rearrange(x1[-1], "c h w -> h w c").to("cpu")
                x1 = wavelet_reconstruction(x1.permute(2, 0, 1), phi2narry_sm(target).permute(0, 3, 1, 2).squeeze(0).to("cpu"))
                x1 = x1.clamp(0, 1)
                img=x1.unsqueeze(0).permute(0, 2, 3, 1) #torch.Size([1, 673, 818, 3])
                if crop_box:
                    original_tensor = phi2narry_sm(original)
                    x1, y1, x2, y2 = crop_box
                    comp_h, comp_w = y2 - y1, x2 - x1
                    composited_resized = torch.nn.functional.interpolate(
                        img.permute(0, 3, 1, 2), 
                        size=(comp_h, comp_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).permute(0, 2, 3, 1)  # [H_crop, W_crop, C]
                    result_tensor = original_tensor.clone()
                    
                    result_rgba= tensor2image_sm(composited_resized[0])
                    transparent_bg.paste(result_rgba,(crop_box[0], crop_box[1]))
                    if save_rgba:
                        transparent_bg.save(os.path.join(folder_paths.get_output_directory(), f"{prefix}.png"))
                    result_tensor[0, y1:y2, x1:x2, :] = composited_resized[0]
                    return result_tensor,transparent_bg
                return img,transparent_bg
            if crop_box:
                result = original.copy()
                result.paste(composited, (crop_box[0], crop_box[1]))
                
                transparent_bg.paste(composited,(crop_box[0], crop_box[1]))
                if save_rgba:
                    transparent_bg.save(os.path.join(folder_paths.get_output_directory(), f"{prefix}.png"))
                return result,transparent_bg
            return composited ,transparent_bg
        result,transparent_bg=paste_back(cond["origin_image"],tensor2image_sm(generated_image),cond["model_mask"],crop_box=cond["crop_box"],mask_grow=mask_grow,blend_blur=blend_blur,wavelet=wavelet, adain=adain,save_rgba=save_rgba) 
          
        return io.NodeOutput( result if wavelet else phi2narry_sm(result),phi2narry_sm(transparent_bg))  
    
class RefineAnything_PreImg(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="RefineAnything_PreImg",
            display_name="RefineAnything_PreImg",
            category="RefineAnything",
            inputs=[
                io.Image.Input("origin_image" ),
                io.Image.Input("mask_image" ),
                io.Boolean.Input("do_focus_crop",default=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Image.Output(display_name="mask_image"),
                io.Conditioning.Output(display_name="cond"),
                    ],
            )
    @classmethod
    
    def execute(cls,origin_image,mask_image,do_focus_crop, ) -> io.NodeOutput:
        def binarise_mask_to_rgb(mask_l: Image.Image) -> Image.Image:
            """Convert an L-mode mask to a clean binary RGB image (spatial condition for the model)."""
            arr = np.where(np.array(mask_l, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
            return Image.fromarray(arr, mode="L").convert("RGB")
        def bbox_from_mask(mask_l: Image.Image) -> tuple[int, int, int, int]:
            """Return tight (x1, y1, x2, y2) bounding box of non-zero pixels."""
            arr = np.array(mask_l, dtype=np.uint8)
            ys, xs = np.where(arr > 0)
            if xs.size == 0:
                raise ValueError("Mask is empty — nothing to refine.")
            w, h = mask_l.size
            return (max(0, int(xs.min())),
                    max(0, int(ys.min())),
                    min(w, int(xs.max()) + 1),
                    min(h, int(ys.max()) + 1))

        origin_image = tensor2image_sm(origin_image)
        mask_l = tensor2image_sm(mask_image).convert("L")
        if mask_l.size != origin_image.size:
            mask_l = mask_l.resize(origin_image.size, Image.NEAREST)

        bbox = bbox_from_mask(mask_l)
        model_image, model_mask = origin_image, mask_l
        crop_box=None
        def focus_crop(
            image: Image.Image,
            mask_l: Image.Image,
            bbox: tuple[int, int, int, int],
            margin: int = 64,
        ) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
            """
            Crop around *bbox* so the diffusion model works on a ~1024² region.

            Returns (cropped_image, cropped_mask, crop_box).
            """
            iw, ih = image.size
            s = math.sqrt(1024 * 1024 / float(iw * ih))
            x1, y1, x2, y2 = bbox

            cx1 = max(0, int(math.floor(max(0.0, x1 * s - margin) / s)))
            cy1 = max(0, int(math.floor(max(0.0, y1 * s - margin) / s)))
            cx2 = min(iw, int(math.ceil(min(iw * s, x2 * s + margin) / s)))
            cy2 = min(ih, int(math.ceil(min(ih * s, y2 * s + margin) / s)))

            crop_box = (cx1, cy1, cx2, cy2)
            return image.crop(crop_box), mask_l.crop(crop_box), crop_box
        if do_focus_crop:
            model_image, model_mask, crop_box = focus_crop(
                origin_image, mask_l, bbox, margin=64,
            )
        cond={"crop_box":crop_box,"model_mask":model_mask,"origin_image":origin_image}
        return io.NodeOutput(phi2narry_sm(model_image),phi2narry_sm(binarise_mask_to_rgb(model_mask)),cond)


class TextEncodeQwenImageEditPlus_NoAppend(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEditPlus_NoAppend",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=False)
        return io.NodeOutput(conditioning)


class RefineAnything_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RefineAnything_PreImg,
            RefineAnything_Pasteback,
            TextEncodeQwenImageEditPlus_NoAppend
        ]   

async def comfy_entrypoint() -> RefineAnything_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return RefineAnything_Extension()


