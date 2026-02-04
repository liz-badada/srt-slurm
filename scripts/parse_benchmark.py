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


def create_pareto_chart(df: pd.DataFrame, title: str = "SGLang DSR1 FP8 H100 Disaggregated 1k1k/1k8k/8k1k (MTP vs non-MTP)") -> go.Figure:
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
    # Create legend with GPU count and seq_len
    # e.g., "h100-fp8-1p2d-max-dep (1k1k)" -> "h100-fp8-1p2d-48gpus-max-dep (1k1k)"
    def create_legend(r):
        config = r['Config']
        gpu_num = int(r['GPU num'])
        
        # Insert GPU count after the XpYd pattern
        # e.g., "h100-fp8-1p2d-max-dep" -> "h100-fp8-1p2d-48gpus-max-dep"
        pd_match = re.search(r'(\d+p\d+d)', config)
        if pd_match:
            pd_part = pd_match.group(1)
            # Insert GPU count right after the pd pattern
            config_with_gpus = config.replace(
                pd_part, 
                f"{pd_part}-{gpu_num}gpus",
                1  # Only replace first occurrence
            )
        else:
            # Fallback: append GPU count at the end of config name
            config_with_gpus = f"{config}-{gpu_num}gpus"
        
        parts = [config_with_gpus]
        if r['seq_len']:
            parts.append(f"({r['seq_len']})")
        return ' '.join(parts)
    
    df['Legend'] = df.apply(create_legend, axis=1)
    
    # Extract parallelism config (e.g., "1p1d-max-dep" from "h100-fp8-1p1d-max-dep-mtp")
    # This is used for color grouping: same parallelism + same ISL/OSL = same color
    def extract_parallelism_config(config):
        """Extract parallelism config without MTP suffix for color grouping."""
        # Remove common prefixes like "h100-fp8-"
        config_clean = re.sub(r'^h\d+-fp\d+-', '', config)
        # Remove -mtp suffix if present
        config_clean = re.sub(r'-mtp$', '', config_clean)
        return config_clean
    
    df['parallelism_config'] = df['Config'].apply(extract_parallelism_config)
    # Color key: parallelism_config + seq_len (same parallelism + same ISL/OSL = same color)
    df['color_key'] = df['parallelism_config'] + '_' + df['seq_len']
    
    # Define marker symbols by PD config
    pd_symbols = {
        '1p1d': 'circle',
        '1p2d': 'square',
        '2p4d': 'diamond',
        '1p4d': 'triangle-up',
        '2p2d': 'cross',
        'unknown': 'star',
    }
    
    # Colorful palette for different configs (easy to distinguish)
    colorful_palette = [
        '#E6194B',  # Red
        '#3CB44B',  # Green
        '#4363D8',  # Blue
        '#F58231',  # Orange
        '#911EB4',  # Purple
        '#42D4F4',  # Cyan
        '#F032E6',  # Magenta
        '#BFEF45',  # Lime
        '#469990',  # Teal
        '#9A6324',  # Brown
        '#800000',  # Maroon
        '#000075',  # Navy
        '#808000',  # Olive
        '#FFE119',  # Yellow
    ]
    
    # Build color mapping: same parallelism + same ISL/OSL = same color
    unique_color_keys = df['color_key'].unique()
    color_map = {key: colorful_palette[i % len(colorful_palette)] 
                 for i, key in enumerate(unique_color_keys)}
    
    # Build figure manually for better control
    fig = go.Figure()
    
    for legend in df['Legend'].unique():
        legend_df = df[df['Legend'] == legend].sort_values('concurrency')
        if legend_df.empty:
            continue
        
        # Get attributes for this legend
        pd_config = legend_df['pd_config'].iloc[0]
        variant = legend_df['variant'].iloc[0]
        color_key = legend_df['color_key'].iloc[0]
        
        # Determine marker symbol based on PD config
        symbol = pd_symbols.get(pd_config, 'circle')
        
        # Determine line style: MTP = solid, non-MTP = dash
        line_style = 'solid' if variant == 'mtp' else 'dash'
        
        # Determine color from color_map (same parallelism + same ISL/OSL = same color)
        color = color_map.get(color_key, '#888888')
        
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
        xaxis=dict(
            dtick=5,  # Finer x-axis tick intervals (every 5 units)
            gridcolor='rgba(128, 128, 128, 0.2)',
            gridwidth=1,
            minor=dict(
                dtick=2.5,  # Minor ticks every 2.5 units
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.1)',
            ),
        ),
        yaxis=dict(
            dtick=5,  # Finer y-axis tick intervals (every 5 units)
            gridcolor='rgba(128, 128, 128, 0.2)',
            gridwidth=1,
            minor=dict(
                dtick=2.5,  # Minor ticks every 2.5 units
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.1)',
            ),
        ),
        legend=dict(
            title=dict(
                text="Label: (concurrency, median TTFT)",
                font=dict(size=11, color="#555555"),
            ),
            x=0.99,
            y=0.99,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1,
            itemwidth=50,  # Make legend line longer
        ),
        hovermode='closest',
        width=1400,  # Wider chart to spread out data points
        height=800,
    )
    
    return fig



def create_pareto_chart_by_seq_len(df: pd.DataFrame, isl: int, osl: int, base_title: str = "SGLang DSR1 FP8 H100 Disaggregated") -> go.Figure:
    """
    Create a Pareto curve chart filtered by specific ISL/OSL.
    
    Args:
        df: Full DataFrame with all benchmark results
        isl: Input sequence length to filter (e.g., 1024, 8192)
        osl: Output sequence length to filter (e.g., 1024, 8192)
        base_title: Base title for the chart
    
    Returns:
        Plotly Figure for the filtered data
    """
    # Filter by ISL/OSL
    filtered_df = df[(df['isl'] == isl) & (df['osl'] == osl)].copy()
    
    if filtered_df.empty:
        return go.Figure()
    
    # Format title with seq len info
    seq_len_str = f"{format_seq_len(isl)}{format_seq_len(osl)}"
    title = f"{base_title} {seq_len_str} (MTP vs non-MTP)"
    
    return create_pareto_chart(filtered_df, title=title)


def main():
    parser = argparse.ArgumentParser(description='Parse benchmark.out files and generate reports')
    parser.add_argument('path', type=str, help='Root path to search for benchmark.out files')
    parser.add_argument('--output', '-o', type=str, default='benchmark_results.html',
                        help='Output HTML file path')
    parser.add_argument('--csv', type=str, help='Save results to CSV file (default: scripts/benchmark_results.csv)')
    parser.add_argument('--no-csv', action='store_true', help='Disable automatic CSV generation')
    parser.add_argument('--title', type=str, default='SGLang DSR1 FP8 H100 Disaggregated 1k1k/1k8k/8k1k (MTP vs non-MTP)',
                        help='Title for the charts')
    parser.add_argument('--png', type=str, help='Save Pareto chart as PNG image file')
    parser.add_argument('--svg', type=str, help='Save main Pareto chart as SVG (default: scripts/pareto.svg)')
    parser.add_argument('--no-svg', action='store_true',
                        help='Disable automatic SVG generation')
    parser.add_argument('--svg-prefix', type=str, default='pareto',
                        help='Prefix for SVG files (default: pareto)')
    
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
    
    # Determine scripts directory (where the script is located)
    script_dir = Path(__file__).parent.resolve()
    
    # Save to CSV by default (unless --no-csv is specified)
    if not args.no_csv:
        if args.csv:
            csv_path = Path(args.csv)
        else:
            csv_path = script_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV to: {csv_path}")
    
    # Create charts
    pareto_fig = create_pareto_chart(df, title=args.title)
    
    # Save Pareto chart as PNG if requested
    if args.png:
        pareto_fig.write_image(args.png)
        print(f"\nSaved Pareto chart (PNG) to: {args.png}")
    
    # Save SVG files by default (unless --no-svg is specified)
    if not args.no_svg:
        # Save main Pareto chart
        if args.svg:
            main_svg_path = Path(args.svg)
        else:
            main_svg_path = script_dir / f"{args.svg_prefix}.svg"
        
        pareto_fig.write_image(str(main_svg_path), format='svg')
        print(f"\nSaved main Pareto chart (SVG) to: {main_svg_path}")
        
        # Generate separate SVG files for each ISL/OSL combination
        seq_combinations = [
            (1024, 1024, '1k1k'),
            (1024, 8192, '1k8k'),
            (8192, 1024, '8k1k'),
        ]
        
        base_title = args.title.split('1k1k')[0].strip() if '1k1k' in args.title else 'SGLang DSR1 FP8 H100 Disaggregated'
        
        for isl, osl, seq_name in seq_combinations:
            # Filter data for this ISL/OSL combination
            seq_df = df[(df['isl'] == isl) & (df['osl'] == osl)]
            
            if seq_df.empty:
                print(f"\nNo data for {seq_name} (ISL={isl}, OSL={osl}), skipping...")
                continue
            
            # Create chart for this combination
            seq_fig = create_pareto_chart_by_seq_len(df, isl, osl, base_title=base_title)
            
            # Save to SVG in scripts directory
            svg_filename = script_dir / f"{args.svg_prefix}_{seq_name}.svg"
            seq_fig.write_image(str(svg_filename), format='svg')
            print(f"Saved Pareto chart for {seq_name} (SVG) to: {svg_filename}")
    
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
        <div id="pareto-chart" style="width:1400px;height:800px;"></div>
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
