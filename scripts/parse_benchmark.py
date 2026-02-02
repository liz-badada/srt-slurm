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
# from plotly.subplots import make_subplots  # Unused - multi-metric chart disabled


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
    isl: int = 0              # input sequence length
    osl: int = 0              # output sequence length
    batch_size: int = 0       # cuda-graph-max-bs
    
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


def extract_config_from_path(filepath: Path) -> tuple[str, str, int, int]:
    """
    Extract config name, framework, GPU count, and batch size from file path.
    
    Expected path patterns:
    - outputs/1234/logs/benchmark.out
    - outputs/1234_configname_xxx/logs/benchmark.out
    
    Returns: (config_name, framework, gpu_num, batch_size)
    """
    parts = filepath.parts
    
    # Default values
    config = "unknown"
    framework = "SGLang"
    gpu_num = 8  # default
    
    # Try to read config.yaml from parent directory
    batch_size = 0
    config_yaml_path = filepath.parent.parent / "config.yaml"
    if config_yaml_path.exists():
        try:
            import yaml
            with open(config_yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            if config_data:
                # Extract config name
                if 'name' in config_data:
                    config = config_data['name']
                # Extract GPU count from resources.prefill_nodes + resources.decode_nodes
                resources = config_data.get('resources', {})
                prefill_nodes = resources.get('prefill_nodes', 1)
                decode_nodes = resources.get('decode_nodes', 1)
                gpus_per_node = resources.get('gpus_per_node', 8)
                gpu_num = (prefill_nodes + decode_nodes) * gpus_per_node
                # Extract batch size from backend.sglang_config.decode
                backend = config_data.get('backend', {})
                sglang_config = backend.get('sglang_config', {})
                decode_config = sglang_config.get('decode', {})
                batch_size = decode_config.get('cuda-graph-max-bs', 0)
        except Exception as e:
            print(f"Warning: Could not parse config.yaml: {e}")
    
    # Fallback: Try to find config from directory name
    if config == "unknown":
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
                r'(h100-[^/]+)',
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
    
    # Only use fallback GPU count inference if config.yaml didn't provide it
    # (i.e., gpu_num is still at default value of 8)
    if gpu_num == 8 and config != "unknown":
        # Try to infer GPU count from config name as fallback
        # Note: This is approximate and may be wrong for some configurations
        pd_match = re.search(r'(\d+)p(\d+)d', config)
        if pd_match:
            # Assume each worker uses 2 nodes (for TP=16 on 8-GPU nodes)
            prefill_workers = int(pd_match.group(1))
            decode_workers = int(pd_match.group(2))
            gpu_num = (prefill_workers + decode_workers) * 2 * 8  # 2 nodes per worker
        elif 'agg' in config:
            gpu_num = 8  # aggregated mode typically uses 1 node
    
    return config, framework, gpu_num, batch_size


def parse_benchmark_file(filepath: Path) -> List[BenchmarkResult]:
    """Parse a benchmark.out file and extract all benchmark results."""
    results = []
    
    config, framework, gpu_num, batch_size = extract_config_from_path(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Extract ISL/OSL from config line (first line usually)
    isl, osl = 0, 0
    for line in lines[:10]:  # Check first 10 lines
        if 'SA-Bench Config:' in line or 'isl=' in line:
            isl_match = re.search(r'isl[=:]\s*(\d+)', line)
            osl_match = re.search(r'osl[=:]\s*(\d+)', line)
            if isl_match:
                isl = int(isl_match.group(1))
            if osl_match:
                osl = int(osl_match.group(1))
            break
    
    current_concurrency = None
    current_output_throughput = None
    current_total_throughput = None
    current_median_ttft = None
    current_median_tpot = None
    is_real_benchmark = False  # distinguish between warmup and actual benchmark
    
    for i, line in enumerate(lines):
        # Check for benchmark run start (actual benchmark with request_rate=inf)
        match = re.search(r'Running benchmark with concurrency:\s*(\d+)', line)
        if match:
            current_concurrency = int(match.group(1))
            is_real_benchmark = True
            continue
        
        # Detect warmup by request_rate (warmup uses request_rate=250.0 or similar)
        # Actual benchmark uses request_rate=inf
        if 'request_rate=inf' in line:
            is_real_benchmark = True
        elif 'request_rate=' in line and 'request_rate=inf' not in line:
            is_real_benchmark = False
        
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
                    isl=isl,
                    osl=osl,
                    batch_size=batch_size,
                ))
            
            # Reset for next block
            current_output_throughput = None
            current_total_throughput = None
            current_median_ttft = None
            current_median_tpot = None
            is_real_benchmark = False  # Reset to avoid carrying state to next block
    
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
            'isl': r.isl,
            'osl': r.osl,
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
        df = df.sort_values(['Framework', 'isl', 'osl', 'Config', 'concurrency'])
    
    return df


def format_seq_len(length: int) -> str:
    """Format sequence length: 1024 -> 1k, 8192 -> 8k, etc."""
    if length >= 1000:
        return f"{length // 1000}k"
    return str(length)


def create_pareto_chart(df: pd.DataFrame, title: str = "SGLang DSR1 FP8 H100 Disaggregated") -> go.Figure:
    """
    Create a Pareto curve chart.
    
    X-axis: otpt/user (output tokens per second per user)
    Y-axis: total tps/gpu (total tokens per second per GPU)
    """
    if df.empty:
        return go.Figure()
    
    # Create legend names with ISL/OSL info (e.g., "config (1k1k)")
    df = df.copy()
    
    # Extract PD config (e.g., "1p2d", "2p4d") and variant (e.g., "mtp", "")
    def extract_pd_config(config):
        """Extract PD configuration like 1p2d, 2p4d from config name."""
        match = re.search(r'(\d+p\d+d)', config)
        return match.group(1) if match else 'unknown'
    
    def extract_variant(config):
        """Extract variant like mtp from config name."""
        if '-mtp' in config:
            return 'mtp'
        elif '-dep' in config:
            return 'dep'
        return 'base'
    
    df['pd_config'] = df['Config'].apply(extract_pd_config)
    df['variant'] = df['Config'].apply(extract_variant)
    df['seq_len'] = df.apply(
        lambda r: f"{format_seq_len(r['isl'])}{format_seq_len(r['osl'])}" if r['isl'] > 0 else '',
        axis=1
    )
    # Create legend with seq_len
    def create_legend(r):
        parts = [r['Config']]
        if r['seq_len']:
            parts.append(f"({r['seq_len']})")
        return ' '.join(parts)
    
    df['Legend'] = df.apply(create_legend, axis=1)
    
    # Define marker symbols by PD config
    pd_symbols = {
        '1p1d': 'circle',
        '1p2d': 'square',
        '2p4d': 'diamond',
        '1p4d': 'triangle-up',
        '2p2d': 'cross',
        'unknown': 'star',
    }
    
    # Define line styles by PD config
    pd_line_styles = {
        '1p1d': 'solid',
        '1p2d': 'dash',
        '2p4d': 'dot',
        '1p4d': 'dashdot',
        '2p2d': 'longdash',
    }
    
    # NVIDIA color scheme
    # MTP variants: Green series (NVIDIA Green)
    # Base variants: Gray/Black series
    mtp_colors = {
        '1k1k': '#76B900',  # NVIDIA Green
        '8k1k': '#8BC34A',  # Light Green
        '1k8k': '#4CAF50',  # Green
        '8k8k': '#2E7D32',  # Dark Green
        '4k4k': '#9CCC65',  # Yellow Green
        '2k2k': '#558B2F',  # Olive Green
    }
    
    base_colors = {
        '1k1k': '#1E90FF',  # Dodger Blue
        '8k1k': '#4169E1',  # Royal Blue
        '1k8k': '#6495ED',  # Cornflower Blue
        '8k8k': '#0000CD',  # Medium Blue
        '4k4k': '#00BFFF',  # Deep Sky Blue
        '2k2k': '#4682B4',  # Steel Blue
    }
    
    # Default fallback colors
    default_mtp_color = '#76B900'  # NVIDIA Green
    default_base_color = '#1E90FF'  # Dodger Blue
    
    # Build figure manually for better control
    fig = go.Figure()
    
    legends_added = set()
    legend_idx = 0
    
    for legend in df['Legend'].unique():
        legend_df = df[df['Legend'] == legend].sort_values('concurrency')
        if legend_df.empty:
            continue
        
        # Get attributes for this legend
        pd_config = legend_df['pd_config'].iloc[0]
        variant = legend_df['variant'].iloc[0]
        seq_len = legend_df['seq_len'].iloc[0]
        
        # Determine marker symbol based on PD config
        symbol = pd_symbols.get(pd_config, 'circle')
        
        # Determine line style based on PD config
        line_style = pd_line_styles.get(pd_config, 'solid')
        
        # Determine color based on variant (MTP vs base) and sequence length
        # MTP: Green series (NVIDIA Green)
        # Base: Gray/Black series
        if variant == 'mtp':
            color = mtp_colors.get(seq_len, default_mtp_color)
        else:
            color = base_colors.get(seq_len, default_base_color)
        
        # Add scatter points with text labels showing (concurrency, TTFT)
        # Format TTFT: show in seconds if >= 1000ms, otherwise in ms
        def format_ttft(ttft_ms):
            if ttft_ms >= 1000:
                return f"{ttft_ms/1000:.1f}s"
            return f"{ttft_ms:.0f}ms"
        
        text_labels = [f"({int(row['concurrency'])}, {format_ttft(row['median TTFT'])})" 
                       for _, row in legend_df.iterrows()]
        
        # Combine markers, lines and text in one trace so legend shows both marker and line style
        fig.add_trace(go.Scatter(
            x=legend_df['otpt/user'],
            y=legend_df['otpt/gpu'],
            mode='lines+markers+text',
            name=legend,
            marker=dict(
                symbol=symbol,
                size=12,
                color=color,
                line=dict(width=1, color=color),
            ),
            line=dict(width=3, color=color, dash=line_style),
            text=text_labels,
            textposition='top center',
            textfont=dict(size=9, color=color),
            hovertemplate=(
                f"<b>{legend}</b><br>"
                "otpt/user: %{x:.1f}<br>"
                "otpt/gpu: %{y:.1f}<br>"
                "<extra></extra>"
            ),
            customdata=legend_df[['concurrency', 'GPU num', 'Output throughput', 'median TTFT', 'median TPOT']].values,
            showlegend=True,
        ))
        
        legend_idx += 1
    
    # Add annotation explaining the point labels (top right corner)
    fig.add_annotation(
        x=0.99,
        y=0.99,
        xref='paper',
        yref='paper',
        text="Label: (concurrency, median TTFT)",
        showarrow=False,
        font=dict(size=11, color="#555555"),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='rgba(0, 0, 0, 0.3)',
        borderwidth=1,
        borderpad=4,
        xanchor='right',
        yanchor='top',
    )
    
    # Update layout with centered title
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title="Output Tokens/s per User",
        yaxis_title="Output Tokens/s per GPU",
        legend=dict(
            x=0.01,
            y=0.01,
            xanchor='left',
            yanchor='bottom',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1,
            itemwidth=50,  # Make legend line longer
        ),
        hovermode='closest',
        width=1200,
        height=800,
    )
    
    return fig



def main():
    parser = argparse.ArgumentParser(description='Parse benchmark.out files and generate reports')
    parser.add_argument('path', type=str, help='Root path to search for benchmark.out files')
    parser.add_argument('--output', '-o', type=str, default='benchmark_results.html',
                        help='Output HTML file path')
    parser.add_argument('--csv', type=str, help='Also save results to CSV file')
    parser.add_argument('--title', type=str, default='SGLang DSR1 FP8 H100 Disaggregated',
                        help='Title for the charts')
    parser.add_argument('--png', type=str, help='Save Pareto chart as PNG image file')
    parser.add_argument('--svg', type=str, help='Save Pareto chart as SVG image file (vector, higher quality)')
    
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
    
    # Save Pareto chart as PNG if requested
    if args.png:
        pareto_fig.write_image(args.png)
        print(f"\nSaved Pareto chart (PNG) to: {args.png}")
    
    # Save Pareto chart as SVG if requested
    if args.svg:
        pareto_fig.write_image(args.svg, format='svg')
        print(f"\nSaved Pareto chart (SVG) to: {args.svg}")
    
    # Save to HTML using JSON (avoid binary encoding issues)
    import json
    pareto_json = pareto_fig.to_json()
    
    # Extract ISL/OSL info from results
    isl_osl_info = ""
    if all_results:
        isl_values = set(r.isl for r in all_results if r.isl > 0)
        osl_values = set(r.osl for r in all_results if r.osl > 0)
        if isl_values or osl_values:
            isl_str = ", ".join(str(v) for v in sorted(isl_values)) if isl_values else "N/A"
            osl_str = ", ".join(str(v) for v in sorted(osl_values)) if osl_values else "N/A"
            isl_osl_info = f"<p><strong>Input Sequence Length (ISL):</strong> {isl_str} &nbsp;&nbsp; <strong>Output Sequence Length (OSL):</strong> {osl_str}</p>"
    
    with open(args.output, 'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>{args.title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 20px 0; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .chart-container {{ margin: 20px 0; }}
        .info-box {{ background-color: #e8f4f8; padding: 10px 15px; border-radius: 5px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h1>{args.title}</h1>
    
    <h2>Summary Table</h2>
    <div class="info-box">
        {isl_osl_info}
    </div>
    {df.to_html(index=False, classes='benchmark-table')}
    
    <h2>Pareto Curve (Output tps/user vs Total tps/gpu)</h2>
    <div class="chart-container">
        <div id="pareto-chart" style="width:1200px;height:800px;"></div>
    </div>
    
    <script>
        var paretoData = {pareto_json};
        Plotly.newPlot('pareto-chart', paretoData.data, paretoData.layout);
    </script>
</body>
</html>
""")
    
    print(f"\nSaved HTML report to: {args.output}")
    print(f"Open in browser: file://{Path(args.output).absolute()}")
    
    return 0


if __name__ == '__main__':
    exit(main())
