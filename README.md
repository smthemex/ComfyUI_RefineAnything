

ComfyUI_RefineAnything
----
[RefineAnything](https://github.com/limuloo/RefineAnything):Multimodal Region-Specific Refinement for Perfect Local Details


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
![](https://github.com/smthemex/ComfyUI_RefineAnything/blob/main/example_workflows/example.png)


## 📖 Citation

```
@article{zhou2026refineanything,
  title={RefineAnything: Multimodal Region-Specific Refinement for Perfect Local Details},
  author={Zhou, Dewei and Li, You and Yang, Zongxin and Yang, Yi},
  journal={arXiv preprint arXiv:2604.06870},
  year={2026}
}
```
