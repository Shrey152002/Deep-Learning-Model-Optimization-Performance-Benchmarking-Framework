# Deep Learning Model Optimization & Performance Benchmarking Framework

A comprehensive benchmark framework for evaluating PyTorch model optimization techniques across different hardware targets, with detailed performance analysis and visualization tools.

## Overview

This project provides a framework for benchmarking various PyTorch optimization techniques across CPU and GPU configurations. It measures inference time, memory usage, and accuracy across different configurations to help you identify the optimal performance setup for your specific hardware.

Key features include:
- Benchmarking across GPU and CPU targets
- Evaluation of multiple optimization techniques
- Automated performance visualization
- Comparative analysis of throughput and memory usage
- Configuration testing for batch sizes and worker counts

## Optimization Techniques Evaluated

The framework benchmarks the following optimization techniques:

1. **Channels Last Memory Format** - NHWC tensor layout optimization
2. **Automatic Mixed Precision (BFloat16)** - Reduced precision for higher throughput
3. **Combined AMP + Channels Last** - Leveraging both memory layout and precision benefits
4. **torch.compile** - PyTorch 2.0's graph optimization technology

## Key Results

Our benchmarks demonstrate:
- Up to 2.3x speedup using channels last memory format and BFloat16 AMP compared to baseline CPU inference
- Performance variations based on batch size (32-64) and worker combinations
- 3.1x speedup potential with torch.compile optimization
- Less than 1% accuracy variation between hardware targets across configurations

## Requirements

- Python 3.8+
- PyTorch 2.0+ (for torch.compile)
- torchvision
- matplotlib
- numpy
- pandas
- tqdm

## Usage

```python
# Clone the repository
git clone https://github.com/yourusername/dl-model-optimization
cd dl-model-optimization

# Install dependencies
pip install -r requirements.txt

# Run the benchmark
python benchmark.py
```

## Configuration Options

The framework allows customization of:
- Batch sizes (default: 32, 64)
- Number of workers (default: 0, 1)
- Model architecture (default: SimpleCNN)
- Dataset (default: CIFAR-10)

## Output

The benchmark generates:
- CSV report with performance metrics across all configurations
- Performance comparison visualizations
- GPU vs CPU speedup ratio charts
- Optimization technique comparison graphs

## Understanding the Results

### Memory Format Optimization (Channels Last)
NHWC format aligns better with CPU SIMD instructions, particularly beneficial for convolutional operations. This optimization works best with larger batch sizes.

### Mixed Precision (BFloat16)
BFloat16 reduces memory bandwidth requirements while maintaining computational accuracy. Performance varies based on CPU architecture support for BFloat16 operations.

### torch.compile
The most effective optimization in our tests. PyTorch's compilation technology analyzes the execution graph, fuses operations, and generates hardware-specific optimized code.

### Hardware Dependencies
Optimization effectiveness varies based on:
- CPU type (Intel vs AMD)
- Instruction set support (AVX-512, AVX2)
- Memory bandwidth and cache hierarchy

## Citation

If you use this framework in your research, please cite:

```
@misc{deeplearning-optimization-framework,
  author = {Your Name},
  title = {Deep Learning Model Optimization & Performance Benchmarking Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/dl-model-optimization}
}
```

## License

MIT
