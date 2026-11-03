GPU: Tesla T4 (7, 5) | dtype: torch.float16
Loading weights: 100% 760/760 [04:20<00:00,  2.92it/s]
kelas model: Qwen3_5ForConditionalGeneration
VRAM setelah muat: 7.38 GiB (267 dtk)
trainable params: 40,108,032 || all params: 9,449,921,776 || trainable%: 0.4244
lapisan terkena LoRA: 0 .. 31 (32 lapisan)

== 768px, stride 2: 14 frame, 6738 token ==
Traceback (most recent call last):
  File "/content/repo/scripts/uji_1_batch.py", line 249, in <module>
    main()
    ~~~~^^
  File "/content/repo/scripts/uji_1_batch.py", line 230, in main
    h = jalankan(model, processor, row, manifest, ms, st, dtype, hemat=not a.logits_penuh)
  File "/content/repo/scripts/uji_1_batch.py", line 113, in jalankan
    loss = hitung_loss(model, batch, n_label, hemat)
  File "/content/repo/scripts/uji_1_batch.py", line 85, in hitung_loss
    logits = model(**masuk, logits_to_keep=n_label + 1).logits[:, :-1]
             ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/peft/peft_model.py", line 2120, in forward
    return self.base_model(
           ~~~~~~~~~~~~~~~^
        input_ids=input_ids,
        ^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/peft/tuners/tuners_utils.py", line 359, in forward
    return self.model(*args, **kwargs)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1850, in forward
    outputs = self.model(
        input_ids=input_ids,
    ...<10 lines>...
        **kwargs,
    )
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1605, in forward
    mm_encoder_outputs["image"] = self.get_image_features(
                                  ~~~~~~~~~~~~~~~~~~~~~~~^
        pixel_values, image_grid_thw, return_dict=True, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/dist-packages/transformers/utils/generic.py", line 1006, in wrapper
    return func(*args, **translated)
  File "/usr/local/lib/python3.13/dist-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1481, in get_image_features
    vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/utils/generic.py", line 1081, in wrapper
    output = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/utils/output_capturing.py", line 287, in wrapper
    outputs = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 1172, in forward
    hidden_states = self.patch_embed(hidden_states)
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py", line 964, in forward
    hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
                    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/conv.py", line 735, in forward
    return self._conv_forward(input, self.weight, self.bias)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/conv.py", line 730, in _conv_forward
    return F.conv3d(
           ~~~~~~~~^
        input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failedptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH

