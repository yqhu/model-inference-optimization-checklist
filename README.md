# (Draft) Model Inference Optimization Checklist

This checklist describes some steps that should be completed when diagnosing model inference performance issues.  Some of these suggestions are only applicable to NLP models (e.g., ensuring the input is not over-padded and sequence bucketing), but the general principles are useful for other models too.

* Check the versions of PyTorch, Nvidia driver, and other components and update to the latest compatible releases.  Oftentimes known performance bugs have already been fixed.  As this example ([using PyTorch 1.1](examples/torch_version_11.py), [using PyTorch 1.9](examples/torch_version_19.py)) shows, the inference time of a Hugging Face bert model reduces by 15% moving from PyTorch 1.1 and Transformers 3.3 to PyTorch 1.9 and Transformers 4.6.

* Collect system-level activity logs to understand the overall resource utilizations. It’s useful to know how the model inference pipeline is using the system resources at a high level, as the first step of optimization.  Even simple CLI tools such as nvidia-smi and htop would be helpful.

* Start with a target with the highest impact on performance.  It should be obvious from the system activity logs where the biggest bottleneck is – look beyond model inference, as pre/post processing can be expensive and can affect the end-to-end throughput just as much.

* Use profilers to zoom into the identified target to further identify the hotspots.  Many profilers can be used, both tracing-based and sampling-based.  Depending on use case on-demand profilers can be used if/when code instrumentation is inconvenient, such as on production code.

* Quantify and mitigate the influence of slow I/O such as disk and network on end-to-end performance.  While optimizing I/O is out of scope for this checklist, look for techniques that use async, concurrency, pipelining, etc. to effectively “hide” the cost of I/O.

* Model inference on input sequences of dynamic length (e.g., transformers for NLP): make sure the tokenizer is not over-padding the input.  If a transformer was trained with padding to a constant length (e.g., 512) and deployed with the same padding, it would run unnecessarily slow (orders of magnitude) on short sequences.

* Vision models with input in JPEG format often benefit from faster JPEG decoding on CPU such as libjpeg-turbo and Pillow-SIMD, and on GPU such as torchvision.io.decode_jpeg and Nvidia DALI.  As [this example](examples/torchvision_vs_dali.py) shows, Nvidia DALI is about 20% faster than torchvision, even on an old K80 GPU.

* Start model inference optimization only after other factors, the “low-hanging fruit”, have been extensively evaluated and addressed.

* Run PyTorch Profiler.  Look at the breakdown of operations by time.  Pay attention to performance suggestions such as increasing num_workers in dataloaders and making use of tensor cores (by utilizing fp16).  Quite often the default num_workers = 0 is starving the GPU by limiting all data loading and preprocessing to the same process; increasing this number to enable multiprocessing would increase GPU utilization drastically.

* Use fp16 for GPU inference.  The speed will most likely more than double on newer GPUs with tensor cores, with negligible accuracy degradation.  Technically fp16 is a type of quantization but since it seldom suffers from loss of accuracy for inference it should always be explored.

* Use model quantization (i.e., int8) for CPU inference.  Explore different quantization options: dynamic quantization, static quantization, and quantization aware training, as well as tools such as Intel Neural Compressor that provide more sophisticated quantization methods.

* Balance throughput and latency with smart batching.  While meeting the latency SLA try larger batch sizes to increase the throughput.

* Try torchscript, inference_mode, and optimize_for_inference.

* Try optimized inference engines such as onnxruntime, tensorRT, lightseq, ctranslate-2, etc.  These engines often provide additional optimizations such as operator fusion, in addition to model quantization.

* Try model distillation.  This is more involved and often requires training data, but the potential gain can be large.  For example, MiniLM achieves 99% the accuracy of the original BERT base model while being 2X faster.

* Try task parallelism.  Python’s GIL could affect effective multithreading, even for external native code.  For a system with 32 vCPUs, two inference sessions each with 16 threads often have higher throughput than a single inference session with 32 threads.  When testing multiple sessions, it is important to set torch.num_threads properly to avoid CPU contention.

* For batch processing on sequences with different lengths, sequence bucketing could potentially improve the throughput by 2X.  In this case, a simple implementation of sequence bucketing is to sort all input by sequence length before feeding them to the model, as this reduces unnecessary padding when batching the sequences.

While this checklist is not exhaustive, going through the items will likely help you squeeze more performance out of your model inference pipeline.  By that time if you still need more optimization, let us know and we would be more than happy to help.
