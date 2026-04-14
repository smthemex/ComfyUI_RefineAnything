

ComfyUI_RefineAnything
----
[RefineAnything](https://github.com/limuloo/RefineAnything):Multimodal Region-Specific Refinement for Perfect Local Details

# Notice
* if use refer image ,refer image must link image2 and mask  link image3；
* 注意，使用参考图模式时，参考图必须连接image2 ，遮罩连image3；


1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
ComfyUI_RefineAnythingComfyUI_MagiHuman
```

2.checkpoints 
-----
* lora [links](https://huggingface.co/limuloo1999/RefineAnything/tree/main) 

```
├── ComfyUI/models/
|     ├── lora/
|        ├──Qwen-Image-Edit-2511-RefineAny.safetensors
|        ├──normal qwen edit 2511 4 steps lora
|     ├── vae/
|        ├──normal qwen edit 2511
|     ├── diffusion_models/ 
|        ├──normal qwen edit 2511
|     ├── clip/ 
|        ├──normal qwen edit 2511

```

3.Example
-----
* normal
![](https://github.com/smthemex/ComfyUI_RefineAnything/blob/main/example_workflows/example.png)
* refer image
![](https://github.com/smthemex/ComfyUI_RefineAnything/blob/main/example_workflows/example_r.png)

## 📖 Citation

```
@article{zhou2026refineanything,
  title={RefineAnything: Multimodal Region-Specific Refinement for Perfect Local Details},
  author={Zhou, Dewei and Li, You and Yang, Zongxin and Yang, Yi},
  journal={arXiv preprint arXiv:2604.06870},
  year={2026}
}
```
