#!/usr/bin/env python3
"""
Parse benchmark.out files and generate summary table + Pareto curve.

Usage:
    python parse_benchmark.py /path/to/outputs
    python parse_benchmark.py /path/to/outputs --output results.html
"""

import argparse
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    config: str
    framework: str
    gpu_num: int
    concurrency: int
    output_throughput: float  # tok/s
    total_throughput: float   # tok/s
    median_ttft: float        # ms
    median_tpot: float        # ms
    
    @property
    def otpt_per_user(self) -> float:
        """Output tokens per second per user (concurrency)."""
        if self.concurrency == 0:
            return 0
        return self.output_throughput / self.concurrency
    
    @property
    def otpt_per_gpu(self) -> float:
        """Output tokens per second per GPU."""
        if self.gpu_num == 0:
            return 0
        return self.output_throughput / self.gpu_num
    
    @property
    def total_tps_per_gpu(self) -> float:
        """Total tokens per second per GPU."""
        if self.gpu_num == 0:
            return 0
        return self.total_throughput / self.gpu_num


def extract_config_from_path(filepath: Path) -> tuple[str, str, int]:
    """
    Extract config name, framework, and GPU count from file path.
    
    Expected path patterns:
    - outputs/1234/logs/benchmark.out
    - outputs/1234_configname_xxx/logs/benchmark.out
    
    Returns: (config_name, framework, gpu_num)
    """
    parts = filepath.parts
    
    # Default values
    config = "unknown"
    framework = "SGLang"
    gpu_num = 8  # default
    
    # Try to find config from directory name
    for part in parts:
        # Check for job directory pattern like "1234_bs128-agg-tp_..."
        if re.match(r'^\d+', part):
            # Extract config name if present
            match = re.search(r'\d+_([^_]+(?:-[^_]+)*)', part)
            if match:
                config = match.group(1)
        
        # Check for config patterns in path
        config_patterns = [
            r'(bs\d+-\d+p\d+d(?:-(?:tp|dep|mtp))?)',
            r'(bs\d+-agg-tp(?:-mtp)?)',
            r'(low-latency-\d+p\d+d)',
            r'(ctx\d+_gen\d+_[^/]+)',
        ]
        for pattern in config_patterns:
            match = re.search(pattern, part)
            if match:
                config = match.group(1)
                break
        
        # Detect framework
        if 'trtllm' in part.lower() or 'trt-llm' in part.lower():
            framework = "TRT-LLM"
        
        # Try to extract GPU count from path
        gpu_match = re.search(r'(\d+)gpu', part.lower())
        if gpu_match:
            gpu_num = int(gpu_match.group(1))
    
    # Try to infer GPU count from config name
    # e.g., bs128-1p1d -> 1 prefill + 1 decode = 2 nodes * 8 gpus = 16 gpus
    pd_match = re.search(r'(\d+)p(\d+)d', config)
    if pd_match:
        prefill_nodes = int(pd_match.group(1))
        decode_nodes = int(pd_match.group(2))
        gpu_num = (prefill_nodes + decode_nodes) * 8
    elif 'agg' in config:
        gpu_num = 8  # aggregated mode typically uses 1 node
    
    return config, framework, gpu_num


def parse_benchmark_file(filepath: Path) -> List[BenchmarkResult]:
    """Parse a benchmark.out file and extract all benchmark results."""
    results = []
    
    config, framework, gpu_num = extract_config_from_path(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by benchmark runs - look for "Running benchmark with concurrency" or results blocks
    # Pattern to match a complete benchmark result block
    pattern = r'(?:Running benchmark with concurrency:\s*(\d+)|Warming up with concurrency\s+(\d+)).*?(?=Running benchmark|Warming up|SA-Bench complete|$)'
    
    # Alternative: parse line by line
    lines = content.split('\n')
    
    current_concurrency = None
    current_output_throughput = None
    current_total_throughput = None
    current_median_ttft = None
    current_median_tpot = None
    is_real_benchmark = False  # distinguish between warmup and actual benchmark
    
    for i, line in enumerate(lines):
        # Check for benchmark run start
        match = re.search(r'Running benchmark with concurrency:\s*(\d+)', line)
        if match:
            current_concurrency = int(match.group(1))
            is_real_benchmark = True
            continue
        
        # Also capture warmup runs if needed
        match = re.search(r'Warming up with concurrency\s+(\d+)', line)
        if match:
            current_concurrency = int(match.group(1))
            is_real_benchmark = False
            continue
        
        # Parse metrics
        if 'Output token throughput (tok/s):' in line:
            match = re.search(r'Output token throughput \(tok/s\):\s*([\d.]+)', line)
            if match:
                current_output_throughput = float(match.group(1))
        
        if 'Total Token throughput (tok/s):' in line:
            match = re.search(r'Total Token throughput \(tok/s\):\s*([\d.]+)', line)
            if match:
                current_total_throughput = float(match.group(1))
        
        if 'Median TTFT (ms):' in line:
            match = re.search(r'Median TTFT \(ms\):\s*([\d.]+)', line)
            if match:
                current_median_ttft = float(match.group(1))
        
        if 'Median TPOT (ms):' in line:
            match = re.search(r'Median TPOT \(ms\):\s*([\d.]+)', line)
            if match:
                current_median_tpot = float(match.group(1))
        
        # End of a benchmark block
        if '==================================================' in line:
            if (current_concurrency is not None and 
                current_output_throughput is not None and
                current_total_throughput is not None and
                current_median_ttft is not None and
                current_median_tpot is not None and
                is_real_benchmark):
                
                results.append(BenchmarkResult(
                    config=config,
                    framework=framework,
                    gpu_num=gpu_num,
                    concurrency=current_concurrency,
                    output_throughput=current_output_throughput,
                    total_throughput=current_total_throughput,
                    median_ttft=current_median_ttft,
                    median_tpot=current_median_tpot,
                ))
            
            # Reset for next block
            current_output_throughput = None
            current_total_throughput = None
            current_median_ttft = None
            current_median_tpot = None
    
    return results


def find_benchmark_files(root_path: Path) -> List[Path]:
    """Find all benchmark.out files under the given path."""
    benchmark_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename == 'benchmark.out':
                benchmark_files.append(Path(dirpath) / filename)
    
    return benchmark_files


def create_summary_table(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Create a summary DataFrame from benchmark results."""
    data = []
    
    for r in results:
        data.append({
            'Framework': r.framework,
            'Config': r.config,
            'GPU num': r.gpu_num,
            'concurrency': r.concurrency,
            'Output throughput': round(r.output_throughput, 2),
            'Total throughput': round(r.total_throughput, 2),
            'median TTFT': round(r.median_ttft, 2),
            'median TPOT': round(r.median_tpot, 2),
            'otpt/user': round(r.otpt_per_user, 2),
            'otpt/gpu': round(r.otpt_per_gpu, 2),
            'total tps/gpu': round(r.total_tps_per_gpu, 2),
        })
    
    df = pd.DataFrame(data)
    
    # Sort by framework, config, concurrency
    if not df.empty:
        df = df.sort_values(['Framework', 'Config', 'concurrency'])
    
    return df


def create_pareto_chart(df: pd.DataFrame, title: str = "SGLang Benchmark Results") -> go.Figure:
    """
    Create a Pareto curve chart.
    
    X-axis: otpt/user (output tokens per second per user)
    Y-axis: total tps/gpu (total tokens per second per GPU)
    """
    if df.empty:
        return go.Figure()
    
    # Create figure
    fig = px.scatter(
        df,
        x='otpt/user',
        y='total tps/gpu',
        color='Config',
        symbol='Framework',
        hover_data=['concurrency', 'GPU num', 'Output throughput', 'median TTFT', 'median TPOT'],
        title=title,
        labels={
            'otpt/user': 'Output Tokens/s per User',
            'total tps/gpu': 'Total Tokens/s per GPU',
        }
    )
    
    # Add lines connecting points for each config
    configs = df['Config'].unique()
    for config in configs:
        config_df = df[df['Config'] == config].sort_values('concurrency')
        if len(config_df) > 1:
            fig.add_trace(go.Scatter(
                x=config_df['otpt/user'],
                y=config_df['total tps/gpu'],
                mode='lines',
                name=f'{config} (line)',
                line=dict(width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="out tps / user",
        yaxis_title="tps/r / gpu",
        legend_title="Config",
        hovermode='closest',
        width=1200,
        height=800,
    )
    
    return fig


def create_multi_metric_chart(df: pd.DataFrame) -> go.Figure:
    """Create a multi-panel chart with different metrics."""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Output Throughput vs Concurrency',
            'Median TTFT vs Concurrency',
            'Median TPOT vs Concurrency',
            'Pareto: otpt/user vs tps/gpu'
        )
    )
    
    configs = df['Config'].unique()
    colors = px.colors.qualitative.Set1
    
    for i, config in enumerate(configs):
        config_df = df[df['Config'] == config].sort_values('concurrency')
        color = colors[i % len(colors)]
        
        # Output throughput vs concurrency
        fig.add_trace(
            go.Scatter(
                x=config_df['concurrency'],
                y=config_df['Output throughput'],
                mode='lines+markers',
                name=config,
                line=dict(color=color),
                legendgroup=config,
            ),
            row=1, col=1
        )
        
        # Median TTFT vs concurrency
        fig.add_trace(
            go.Scatter(
                x=config_df['concurrency'],
                y=config_df['median TTFT'],
                mode='lines+markers',
                name=config,
                line=dict(color=color),
                legendgroup=config,
                showlegend=False,
            ),
            row=1, col=2
        )
        
        # Median TPOT vs concurrency
        fig.add_trace(
            go.Scatter(
                x=config_df['concurrency'],
                y=config_df['median TPOT'],
                mode='lines+markers',
                name=config,
                line=dict(color=color),
                legendgroup=config,
                showlegend=False,
            ),
            row=2, col=1
        )
        
        # Pareto curve
        fig.add_trace(
            go.Scatter(
                x=config_df['otpt/user'],
                y=config_df['total tps/gpu'],
                mode='lines+markers',
                name=config,
                line=dict(color=color),
                legendgroup=config,
                showlegend=False,
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=900,
        width=1400,
        title_text="Benchmark Results Overview",
        showlegend=True,
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Concurrency", row=1, col=1)
    fig.update_xaxes(title_text="Concurrency", row=1, col=2)
    fig.update_xaxes(title_text="Concurrency", row=2, col=1)
    fig.update_xaxes(title_text="out tps / user", row=2, col=2)
    
    fig.update_yaxes(title_text="Output tok/s", row=1, col=1)
    fig.update_yaxes(title_text="TTFT (ms)", row=1, col=2)
    fig.update_yaxes(title_text="TPOT (ms)", row=2, col=1)
    fig.update_yaxes(title_text="tps/r / gpu", row=2, col=2)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Parse benchmark.out files and generate reports')
    parser.add_argument('path', type=str, help='Root path to search for benchmark.out files')
    parser.add_argument('--output', '-o', type=str, default='benchmark_results.html',
                        help='Output HTML file path')
    parser.add_argument('--csv', type=str, help='Also save results to CSV file')
    parser.add_argument('--title', type=str, default='SGLang Benchmark Results',
                        help='Title for the charts')
    
    args = parser.parse_args()
    
    root_path = Path(args.path)
    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist")
        return 1
    
    # Find all benchmark files
    print(f"Searching for benchmark.out files in {root_path}...")
    benchmark_files = find_benchmark_files(root_path)
    print(f"Found {len(benchmark_files)} benchmark.out files")
    
    if not benchmark_files:
        print("No benchmark.out files found")
        return 1
    
    # Parse all files
    all_results = []
    for filepath in benchmark_files:
        print(f"Parsing: {filepath}")
        results = parse_benchmark_file(filepath)
        print(f"  Found {len(results)} benchmark results")
        all_results.extend(results)
    
    print(f"\nTotal benchmark results: {len(all_results)}")
    
    if not all_results:
        print("No benchmark results found")
        return 1
    
    # Create summary table
    df = create_summary_table(all_results)
    
    # Print table to console
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # Save to CSV if requested
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV to: {args.csv}")
    
    # Create charts
    pareto_fig = create_pareto_chart(df, title=args.title)
    multi_fig = create_multi_metric_chart(df)
    
    # Save to HTML
    with open(args.output, 'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>{args.title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .chart-container {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>{args.title}</h1>
    
    <h2>Summary Table</h2>
    {df.to_html(index=False, classes='benchmark-table')}
    
    <h2>Pareto Curve</h2>
    <div class="chart-container">
        {pareto_fig.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    
    <h2>Multi-Metric Overview</h2>
    <div class="chart-container">
        {multi_fig.to_html(full_html=False, include_plotlyjs=False)}
    </div>
</body>
</html>
""")
    
    print(f"\nSaved HTML report to: {args.output}")
    print(f"Open in browser: file://{Path(args.output).absolute()}")
    
    return 0


if __name__ == '__main__':
    exit(main())
