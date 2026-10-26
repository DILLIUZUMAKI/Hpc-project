import json
import math
import statistics
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .storage import rows_for

# NVIDIA Obsidian & Emerald Green High-Performance Palette
COLOR_SERIAL = '#94A3B8'      # Muted Slate (Serial Baseline)
COLOR_REFERENCE = '#38BDF8'   # Sky Blue (Curated Reference)
COLOR_AI_BASE = '#F59E0B'     # Amber (AI Baseline)
COLOR_AI_ACCEPTED = '#10B981' # NVIDIA Emerald (AI Accepted Iteration)
COLOR_AI_REJECTED = '#F43F5E' # Rose (AI Rejected Iteration)
COLOR_IDEAL = '#64748B'       # Subtle Slate (Ideal Linear Scaling)
COLOR_EMERALD = '#10B981'
COLOR_COBALT = '#10B981'     # Primary NVIDIA Emerald
COLOR_MINT = '#34D399'        # High-contrast Mint Accent
COLOR_TEAL = '#059669'        # Deep Emerald/Teal
COLOR_AMBER = '#F59E0B'       # Amber Accent
COLOR_SURFACE = '#131920'
COLOR_BORDER = '#1A232F'

COLORS = [COLOR_EMERALD, COLOR_REFERENCE, COLOR_AI_BASE, COLOR_MINT, COLOR_SERIAL, COLOR_AI_REJECTED]

CANDIDATE_COLORS = {
    'serial': COLOR_SERIAL,
    'reference': COLOR_REFERENCE,
    'ai_baseline': COLOR_AI_BASE,
}


def candidate_color(ident, origin='', accepted=True):
    if ident == 'serial' or origin == 'curated_serial':
        return COLOR_SERIAL
    elif ident == 'reference' or origin == 'curated_reference':
        return COLOR_REFERENCE
    elif ident == 'ai_baseline' or origin == 'llm_one_shot':
        return COLOR_AI_BASE
    elif 'iteration' in ident or origin == 'llm_feedback':
        return COLOR_AI_ACCEPTED if accepted else COLOR_AI_REJECTED
    return COLOR_EMERALD


def table(experiment):
    return pd.DataFrame(rows_for(experiment))


def apply_hpc_layout(fig, title='', height=370, hovermode='closest'):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': '"JetBrains Mono", monospace', 'color': '#E2E8F0', 'size': 11.5},
        margin={'l': 50, 'r': 25, 't': 45 if title else 30, 'b': 45},
        height=height,
        hovermode=hovermode,
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'left',
            'x': 0,
            'bgcolor': 'rgba(0,0,0,0)',
            'bordercolor': 'rgba(0,0,0,0)',
            'font': {'size': 11, 'color': '#94A3B8'}
        },
        xaxis={
            'gridcolor': '#16202C',
            'linecolor': '#243346',
            'zerolinecolor': '#16202C',
            'showgrid': True,
            'tickfont': {'size': 11, 'color': '#94A3B8'},
            'title_font': {'size': 11.5, 'color': '#CBD5E1'}
        },
        yaxis={
            'gridcolor': '#16202C',
            'linecolor': '#243346',
            'zerolinecolor': '#16202C',
            'showgrid': True,
            'tickfont': {'size': 11, 'color': '#94A3B8'},
            'title_font': {'size': 11.5, 'color': '#CBD5E1'}
        },
        hoverlabel={
            'bgcolor': '#131920',
            'bordercolor': '#10B981',
            'font': {'family': '"JetBrains Mono", monospace', 'size': 11.5, 'color': '#F8FAFC'}
        }
    )
    if title:
        fig.update_layout(title={'text': title, 'font': {'size': 13, 'color': '#F8FAFC', 'family': '"JetBrains Mono", monospace'}, 'x': 0, 'y': 0.98})
    return fig


def chart(experiment, kind):
    if not experiment:
        return None

    # Handle Variance & Jitter Box / Strip plot
    if kind in {'variance', 'jitter', 'distribution'}:
        fig_var = go.Figure()
        has_points = False
        for cand in experiment.get('candidates', []):
            if cand.get('status') != 'valid':
                continue
            c_id = cand['id']
            col = candidate_color(c_id, cand.get('origin', ''), cand.get('accepted', True))
            for m in cand.get('measurements', []):
                w = m['workers']
                samples = m.get('samples_seconds', [])
                if samples:
                    has_points = True
                    fig_var.add_trace(go.Box(
                        y=samples,
                        name=f"{c_id} (p={w})",
                        marker_color=col,
                        boxpoints='all',
                        jitter=0.35,
                        pointpos=-1.8,
                        marker=dict(size=5, opacity=0.85),
                        line=dict(width=1.5),
                        hovertemplate='<b>%{fullData.name}</b><br>Sample: %{y:.6g}s<extra></extra>'
                    ))
        if not has_points:
            return None
        fig_var.update_layout(yaxis_title='Wall Time (s)', xaxis_title='Implementation & Worker Configuration')
        return apply_hpc_layout(fig_var, 'Run-to-Run Timing Variance & Measurement Jitter')

    # Handle Optimization Trajectory / Waterfall
    if kind in {'iterations', 'trajectory', 'waterfall'}:
        candidates = experiment.get('candidates', [])
        if not candidates:
            return None
        
        # Build candidate progression
        points = []
        for c in candidates:
            c_id = c['id']
            origin = c.get('origin', '')
            status = c.get('status', 'unknown')
            meas = c.get('measurements', [])
            accepted = c.get('accepted', True)
            decision = c.get('decision', '')
            
            score = None
            if status == 'valid' and meas:
                meds = [m['median_seconds'] for m in meas if m.get('median_seconds')]
                if meds:
                    score = math.exp(statistics.mean(math.log(m) for m in meds))
            
            points.append({
                'id': c_id,
                'origin': origin,
                'status': status,
                'score': score,
                'accepted': accepted,
                'decision': decision,
                'is_ai': origin.startswith('llm') or 'ai' in c_id
            })
        
        valid_points = [p for p in points if p['score'] is not None]
        if not valid_points:
            return None

        fig = go.Figure()
        
        # Trajectory path connecting all measured points
        fig.add_trace(go.Scatter(
            x=[p['id'] for p in valid_points],
            y=[p['score'] for p in valid_points],
            mode='lines',
            line=dict(color='#2D3B4E', width=2, dash='dot'),
            name='Optimization Trajectory',
            hoverinfo='skip'
        ))
        
        acc_x, acc_y, acc_colors, acc_text = [], [], [], []
        rej_x, rej_y, rej_text = [], [], []
        
        for p in valid_points:
            hover_desc = f"<br>Rationale: {p['decision']}" if p['decision'] else ''
            if p['id'] == 'serial':
                acc_x.append(p['id']); acc_y.append(p['score']); acc_colors.append(COLOR_SERIAL)
                acc_text.append(f"<b>Serial C Baseline</b><br>Geo-Mean: {p['score']:.6g}s{hover_desc}")
            elif p['id'] == 'reference':
                acc_x.append(p['id']); acc_y.append(p['score']); acc_colors.append(COLOR_REFERENCE)
                acc_text.append(f"<b>Curated Reference</b><br>Geo-Mean: {p['score']:.6g}s{hover_desc}")
            elif p['id'] == 'ai_baseline':
                acc_x.append(p['id']); acc_y.append(p['score']); acc_colors.append(COLOR_AI_BASE)
                acc_text.append(f"<b>AI Baseline (One-Shot)</b><br>Geo-Mean: {p['score']:.6g}s{hover_desc}")
            elif p['accepted']:
                acc_x.append(p['id']); acc_y.append(p['score']); acc_colors.append(COLOR_AI_ACCEPTED)
                acc_text.append(f"<b>{p['id']} (Accepted)</b><br>Geo-Mean: {p['score']:.6g}s{hover_desc}")
            else:
                rej_x.append(p['id']); rej_y.append(p['score'])
                rej_text.append(f"<b>{p['id']} (Rejected)</b><br>Geo-Mean: {p['score']:.6g}s{hover_desc}")

        if acc_x:
            fig.add_trace(go.Scatter(
                x=acc_x,
                y=acc_y,
                mode='markers+text',
                name='Accepted / Baselines',
                marker=dict(size=12, color=acc_colors, symbol='circle', line=dict(color='#080B0E', width=2)),
                text=[f"{y:.4f}s" for y in acc_y],
                textposition='top center',
                textfont=dict(size=10, color='#CBD5E1', family='"JetBrains Mono", monospace'),
                hovertext=acc_text,
                hoverinfo='text'
            ))

        if rej_x:
            fig.add_trace(go.Scatter(
                x=rej_x,
                y=rej_y,
                mode='markers+text',
                name='Rejected Revisions',
                marker=dict(size=12, color=COLOR_AI_REJECTED, symbol='circle-open', line=dict(color=COLOR_AI_REJECTED, width=2.5)),
                text=[f"{y:.4f}s" for y in rej_y],
                textposition='bottom center',
                textfont=dict(size=10, color=COLOR_AI_REJECTED, family='"JetBrains Mono", monospace'),
                hovertext=rej_text,
                hoverinfo='text'
            ))
            
        fig.update_layout(
            yaxis_title='Geometric Mean Runtime (s) [Lower is Better]',
            xaxis_title='Optimization Progression'
        )
        return apply_hpc_layout(fig, 'Candidate Progression & Selection Trajectory')

    # Handle Kernel vs Communication Time Breakdown
    if kind in {'profile_breakdown', 'breakdown'}:
        fig_breakdown = go.Figure()
        has_breakdown = False
        for cand in experiment.get('candidates', []):
            meas = [m for m in cand.get('measurements', []) if m.get('reported_median_kernel_seconds') is not None]
            if meas:
                x_labels = [f"{cand['id']} (p={m['workers']})" for m in meas]
                k_times = [m['reported_median_kernel_seconds'] for m in meas]
                c_times = [m.get('reported_median_communication_seconds', 0) for m in meas]
                fig_breakdown.add_trace(go.Bar(
                    name=f"{cand['id']} Kernel",
                    x=x_labels,
                    y=k_times,
                    marker_color=COLOR_EMERALD,
                    hovertemplate='<b>Kernel Compute</b>: %{y:.6g}s<extra></extra>'
                ))
                fig_breakdown.add_trace(go.Bar(
                    name=f"{cand['id']} Comm/Sync",
                    x=x_labels,
                    y=c_times,
                    marker_color=COLOR_AMBER,
                    hovertemplate='<b>Communication & Sync</b>: %{y:.6g}s<extra></extra>'
                ))
                has_breakdown = True
        if not has_breakdown:
            return None
        fig_breakdown.update_layout(barmode='stack', yaxis_title='Reported Process Time (s)')
        return apply_hpc_layout(fig_breakdown, 'Kernel vs Communication Time Breakdown')

    # Handle Hardware Profile Counters (IPC & Cache Miss Rate %)
    if kind in {'profile_counters', 'profile_ipc', 'hardware'}:
        data = []
        for c in experiment.get('candidates', []):
            prof = c.get('profile', {})
            counters = prof.get('counters', {})
            ipc, miss_rate = None, None
            if counters.get('instructions') and counters.get('cycles'):
                ipc = counters['instructions'] / counters['cycles']
            if counters.get('cache-references') and counters.get('cache-misses') is not None:
                miss_rate = (counters['cache-misses'] / counters['cache-references']) * 100
            if prof.get('kind') == 'cachegrind' and counters:
                total_refs = counters.get('Ir', 0) + counters.get('Dr', 0) + counters.get('Dw', 0)
                total_misses = counters.get('ILmr', 0) + counters.get('DLmr', 0) + counters.get('DLmw', 0)
                if total_refs > 0:
                    miss_rate = (total_misses / total_refs) * 100
            if ipc is not None or miss_rate is not None:
                data.append({'id': c['id'], 'ipc': ipc, 'miss_rate': miss_rate})
        if not data:
            return None
        fig_prof = go.Figure()
        names = [d['id'] for d in data]
        ipcs = [d['ipc'] for d in data]
        misses = [d['miss_rate'] for d in data]
        if any(x is not None for x in ipcs):
            fig_prof.add_trace(go.Bar(
                name='Instructions / Cycle (IPC)',
                x=names,
                y=[x if x is not None else 0 for x in ipcs],
                marker_color=COLOR_EMERALD,
                hovertemplate='<b>%{x}</b><br>IPC: %{y:.2f}<extra></extra>'
            ))
        if any(x is not None for x in misses):
            fig_prof.add_trace(go.Bar(
                name='Cache Miss Rate (%)',
                x=names,
                y=[x if x is not None else 0 for x in misses],
                marker_color=COLOR_AMBER,
                hovertemplate='<b>%{x}</b><br>Cache Miss Rate: %{y:.2f}%<extra></extra>'
            ))
        fig_prof.update_layout(barmode='group', yaxis_title='Metric Value')
        return apply_hpc_layout(fig_prof, 'Hardware Performance Counters Comparison')

    # Standard tabular charts
    df = table(experiment)
    if df.empty:
        return None

    labels = {
        'workers': 'Threads / MPI processes',
        'median_seconds': 'Median process wall time (s)',
        'speedup': 'Speedup vs serial C',
        'efficiency': 'Parallel efficiency',
        'candidate': 'Implementation'
    }

    if kind == 'runtime':
        fig = go.Figure()
        for cand in df['candidate'].unique():
            cdf = df[df.candidate == cand].sort_values('workers')
            c_meta = next((c for c in experiment.get('candidates', []) if c['id'] == cand), {})
            col = candidate_color(cand, c_meta.get('origin', ''), c_meta.get('accepted', True))
            fig.add_trace(go.Bar(
                name=cand,
                x=cdf['workers'],
                y=cdf['median_seconds'],
                error_y=dict(type='data', array=cdf['stdev_seconds'], visible=True, color='#64748B'),
                marker_color=col,
                hovertemplate='<b>%{fullData.name}</b><br>Workers: %{x}<br>Median Wall Time: %{y:.6g}s<br>StdDev: ?%{error_y.array:.4g}s<extra></extra>'
            ))
        fig.update_layout(barmode='group', xaxis_title=labels['workers'], yaxis_title=labels['median_seconds'])
        return apply_hpc_layout(fig, 'Median Process Wall Time (s)')

    elif kind == 'speedup':
        parallel = df[df.candidate != 'serial']
        if parallel.empty:
            return None
        workers = sorted(parallel['workers'].unique())
        fig = go.Figure()
        
        # Add Ideal Linear Reference Line (S = p)
        if workers:
            min_w = min(workers)
            max_w = max(workers)
            fig.add_trace(go.Scatter(
                x=[min_w, max_w],
                y=[min_w, max_w],
                mode='lines',
                line=dict(dash='dash', color=COLOR_IDEAL, width=1.5),
                name='Ideal Linear (S = p)',
                hoverinfo='skip'
            ))
            
        for cand in parallel['candidate'].unique():
            cdf = parallel[parallel.candidate == cand].sort_values('workers')
            c_meta = next((c for c in experiment.get('candidates', []) if c['id'] == cand), {})
            col = candidate_color(cand, c_meta.get('origin', ''), c_meta.get('accepted', True))
            fig.add_trace(go.Scatter(
                x=cdf['workers'],
                y=cdf['speedup'],
                mode='lines+markers',
                name=cand,
                line=dict(color=col, width=2.5),
                marker=dict(size=8, color=col),
                customdata=list(zip(cdf['efficiency'], cdf['median_seconds'])),
                hovertemplate='<b>%{fullData.name}</b><br>Threads/Ranks: %{x}<br>Speedup: %{y:.2f}x vs serial<br>Efficiency: %{customdata[0]:.1%}<br>Wall Time: %{customdata[1]:.4f}s<extra></extra>'
            ))
        fig.update_layout(xaxis_title=labels['workers'], yaxis_title='Speedup vs Serial C (?)')
        return apply_hpc_layout(fig, 'Parallel Speedup vs Ideal Linear Scaling')

    elif kind == 'efficiency':
        parallel = df[df.candidate != 'serial']
        if parallel.empty:
            return None
        fig = go.Figure()
        
        # Add 100% ideal efficiency reference
        fig.add_hline(y=1.0, line_dash='dash', line_color=COLOR_IDEAL,
                      annotation_text='Ideal (100%)', annotation_position='top left',
                      annotation_font=dict(size=10, color=COLOR_IDEAL))
        # Add 50% warning threshold reference
        fig.add_hline(y=0.5, line_dash='dot', line_color='#F59E0B',
                      annotation_text='Low Efficiency Threshold (50%)', annotation_position='bottom left',
                      annotation_font=dict(size=10, color='#F59E0B'))
                      
        for cand in parallel['candidate'].unique():
            cdf = parallel[parallel.candidate == cand].sort_values('workers')
            c_meta = next((c for c in experiment.get('candidates', []) if c['id'] == cand), {})
            col = candidate_color(cand, c_meta.get('origin', ''), c_meta.get('accepted', True))
            fig.add_trace(go.Scatter(
                x=cdf['workers'],
                y=cdf['efficiency'],
                mode='lines+markers',
                name=cand,
                line=dict(color=col, width=2.5),
                marker=dict(size=8, color=col),
                customdata=list(zip(cdf['speedup'], cdf['median_seconds'])),
                hovertemplate='<b>%{fullData.name}</b><br>Threads/Ranks: %{x}<br>Efficiency: %{y:.1%}<br>Speedup: %{customdata[0]:.2f}x<br>Wall Time: %{customdata[1]:.4f}s<extra></extra>'
            ))
        fig.update_layout(xaxis_title=labels['workers'], yaxis_title='Parallel Efficiency (Speedup / p)', yaxis_tickformat='.0%')
        return apply_hpc_layout(fig, 'Parallel Efficiency by Worker Count')

    elif kind == 'scaling':
        parallel = df[df.candidate != 'serial']
        if parallel.empty:
            return None
        fig = go.Figure()
        for cand in parallel['candidate'].unique():
            cdf = parallel[parallel.candidate == cand].sort_values('workers')
            c_meta = next((c for c in experiment.get('candidates', []) if c['id'] == cand), {})
            col = candidate_color(cand, c_meta.get('origin', ''), c_meta.get('accepted', True))
            fig.add_trace(go.Scatter(
                x=cdf['workers'],
                y=cdf['median_seconds'],
                mode='lines+markers',
                name=cand,
                line=dict(color=col, width=2.5),
                marker=dict(size=8, color=col),
                hovertemplate='<b>%{fullData.name}</b><br>Threads/Ranks: %{x}<br>Median Wall Time: %{y:.6g}s<extra></extra>'
            ))
        fig.update_layout(xaxis_title=labels['workers'], yaxis_title=labels['median_seconds'])
        return apply_hpc_layout(fig, 'Strong Scaling: Wall Time vs Thread Count')

    return None
