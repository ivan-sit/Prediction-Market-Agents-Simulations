"""D3.js-based HTML animation renderer.

Generates self-contained HTML files with interactive network visualizations
of information flow in prediction market simulations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .animation_exporter import AnimationExporter


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Information Flow Animation - {run_id}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }}
        #container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        #header {{
            padding: 15px 20px;
            background: #16213e;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }}
        #header h1 {{
            font-size: 1.2rem;
            font-weight: 500;
        }}
        #stats {{
            display: flex;
            gap: 30px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #e94560;
        }}
        .stat-label {{
            font-size: 0.75rem;
            color: #888;
            text-transform: uppercase;
        }}
        #main {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}
        #graph-container {{
            flex: 1;
            position: relative;
        }}
        #graph {{
            width: 100%;
            height: 100%;
        }}
        #sidebar {{
            width: 300px;
            background: #16213e;
            border-left: 1px solid #0f3460;
            padding: 20px;
            overflow-y: auto;
        }}
        #sidebar h2 {{
            font-size: 0.9rem;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 15px;
        }}
        .event-item {{
            background: #1a1a2e;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 3px solid #e94560;
        }}
        .event-type {{
            font-size: 0.7rem;
            text-transform: uppercase;
            color: #e94560;
            margin-bottom: 5px;
        }}
        .event-content {{
            font-size: 0.85rem;
            color: #ccc;
        }}
        #timeline {{
            height: 120px;
            background: #16213e;
            border-top: 1px solid #0f3460;
            padding: 10px 20px;
        }}
        #controls {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 10px;
        }}
        button {{
            background: #e94560;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
        }}
        button:hover {{
            background: #ff6b6b;
        }}
        button:disabled {{
            background: #444;
            cursor: not-allowed;
        }}
        #speed-control {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        #speed-slider {{
            width: 100px;
        }}
        #timeline-chart {{
            height: 60px;
        }}
        .node {{
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .node:hover {{
            filter: brightness(1.2);
        }}
        .node-label {{
            font-size: 10px;
            fill: #fff;
            text-anchor: middle;
            pointer-events: none;
        }}
        .link {{
            stroke-opacity: 0.4;
            transition: stroke-opacity 0.3s ease;
        }}
        .link:hover {{
            stroke-opacity: 1;
        }}
        .particle {{
            fill: #e94560;
            opacity: 0.8;
        }}
        .legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(22, 33, 62, 0.9);
            padding: 15px;
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
            font-size: 0.8rem;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 0.85rem;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.3); opacity: 0.7; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        .pulsing {{
            animation: pulse 0.5s ease-in-out;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="header">
            <h1>Information Flow Animation - {run_id}</h1>
            <div id="stats">
                <div class="stat">
                    <div class="stat-value" id="timestep-display">0</div>
                    <div class="stat-label">Timestep</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="price-display">0.50</div>
                    <div class="stat-label">Market Price</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="events-display">0</div>
                    <div class="stat-label">Events</div>
                </div>
            </div>
        </div>
        <div id="main">
            <div id="graph-container">
                <svg id="graph"></svg>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: linear-gradient(to right, #d73027, #f7f7f7, #1a9850);"></div>
                        <span>Belief: NO (red) → YES (green)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #1DA1F2;"></div>
                        <span>Twitter</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #E74C3C;"></div>
                        <span>Reuters/News</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #9B59B6;"></div>
                        <span>Insider</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #2ECC71;"></div>
                        <span>Analyst Report</span>
                    </div>
                </div>
            </div>
            <div id="sidebar">
                <h2>Events This Timestep</h2>
                <div id="event-list"></div>
            </div>
        </div>
        <div id="timeline">
            <div id="controls">
                <button id="play-btn">Play</button>
                <button id="step-back-btn">&lt;</button>
                <button id="step-forward-btn">&gt;</button>
                <button id="reset-btn">Reset</button>
                <div id="speed-control">
                    <label>Speed:</label>
                    <input type="range" id="speed-slider" min="0.5" max="3" step="0.5" value="1">
                    <span id="speed-value">1x</span>
                </div>
            </div>
            <svg id="timeline-chart"></svg>
        </div>
    </div>
    <div class="tooltip" id="tooltip" style="display: none;"></div>

    <script>
        // Animation data
        const animationData = {animation_data};

        // State
        let currentTimestep = 0;
        let isPlaying = false;
        let playInterval = null;
        let speed = 1;

        // D3 selections
        let svg, simulation, nodeGroup, linkGroup, particleGroup;
        let nodes, links;

        // Belief color scale (red = low, yellow = mid, green = high)
        const beliefColorScale = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);

        // Initialize
        function init() {{
            setupGraph();
            setupTimeline();
            setupControls();
            updateFrame(0);
        }}

        function setupGraph() {{
            const container = document.getElementById('graph-container');
            const width = container.clientWidth;
            const height = container.clientHeight;

            svg = d3.select('#graph')
                .attr('width', width)
                .attr('height', height);

            // Define arrow markers for edges
            svg.append('defs').append('marker')
                .attr('id', 'arrowhead')
                .attr('viewBox', '-0 -5 10 10')
                .attr('refX', 20)
                .attr('refY', 0)
                .attr('orient', 'auto')
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .append('path')
                .attr('d', 'M 0,-5 L 10,0 L 0,5')
                .attr('fill', '#666');

            // Create groups
            linkGroup = svg.append('g').attr('class', 'links');
            particleGroup = svg.append('g').attr('class', 'particles');
            nodeGroup = svg.append('g').attr('class', 'nodes');

            // Prepare data
            nodes = animationData.network.nodes.map(n => ({{...n}}));
            links = animationData.network.edges.map(e => ({{
                source: e.source,
                target: e.target,
                type: e.type
            }}));

            // Create force simulation with layered layout
            simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('y', d3.forceY(d => {{
                    if (d.type === 'source') return height * 0.2;
                    if (d.type === 'agent') return height * 0.5;
                    return height * 0.8;
                }}).strength(0.5))
                .force('collision', d3.forceCollide().radius(d => (d.radius || 20) + 10));

            // Draw links
            const linkElements = linkGroup.selectAll('line')
                .data(links)
                .enter()
                .append('line')
                .attr('class', 'link')
                .attr('stroke', d => d.type === 'subscription' ? '#4a5568' : '#F39C12')
                .attr('stroke-width', d => d.type === 'subscription' ? 1 : 2)
                .attr('stroke-dasharray', d => d.type === 'subscription' ? '5,5' : 'none')
                .attr('marker-end', 'url(#arrowhead)');

            // Draw nodes
            const nodeElements = nodeGroup.selectAll('g')
                .data(nodes)
                .enter()
                .append('g')
                .attr('class', 'node')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended))
                .on('mouseover', showTooltip)
                .on('mouseout', hideTooltip);

            // Source node colors
            const sourceColors = {{
                'twitter': '#1DA1F2',
                'reuters': '#E74C3C',
                'insider': '#9B59B6',
                'analyst_report': '#2ECC71',
                'news_feed': '#E74C3C',
                'expert_analysis': '#2ECC71'
            }};

            nodeElements.append('circle')
                .attr('r', d => {{
                    if (d.type === 'market') return 30;
                    if (d.type === 'agent') return 25;
                    return 20;
                }})
                .attr('fill', d => {{
                    if (d.type === 'source') return sourceColors[d.id] || '#9B59B6';
                    if (d.type === 'market') return '#F39C12';
                    return '#3498DB';
                }})
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);

            // Add value text inside nodes (for agents and market)
            nodeElements.append('text')
                .attr('class', 'node-value')
                .attr('dy', 4)
                .attr('text-anchor', 'middle')
                .attr('fill', '#fff')
                .attr('font-size', '11px')
                .attr('font-weight', 'bold')
                .text(d => {{
                    if (d.type === 'market') return '0.50';
                    if (d.type === 'agent') return '0.50';
                    return '';
                }});

            // Add label below node
            nodeElements.append('text')
                .attr('class', 'node-label')
                .attr('dy', d => {{
                    if (d.type === 'market') return 45;
                    if (d.type === 'agent') return 40;
                    return 35;
                }})
                .text(d => d.label || d.id);

            // Add position text below label (for agents only)
            nodeElements.filter(d => d.type === 'agent')
                .append('text')
                .attr('class', 'node-position')
                .attr('dy', 54)
                .attr('text-anchor', 'middle')
                .attr('fill', '#888')
                .attr('font-size', '9px')
                .text('Y:0 N:0');

            // Update positions on tick
            simulation.on('tick', () => {{
                linkElements
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                nodeElements.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
            }});
        }}

        function setupTimeline() {{
            const container = document.getElementById('timeline-chart');
            const width = container.parentElement.clientWidth - 40;
            const height = 60;

            const timelineSvg = d3.select('#timeline-chart')
                .attr('width', width)
                .attr('height', height);

            const prices = animationData.timeline.prices;
            if (prices.length === 0) return;

            const xScale = d3.scaleLinear()
                .domain([0, prices.length - 1])
                .range([30, width - 10]);

            const yScale = d3.scaleLinear()
                .domain([0, 1])
                .range([height - 10, 10]);

            // Price line
            const line = d3.line()
                .x((d, i) => xScale(i))
                .y(d => yScale(d))
                .curve(d3.curveMonotoneX);

            timelineSvg.append('path')
                .datum(prices)
                .attr('fill', 'none')
                .attr('stroke', '#e94560')
                .attr('stroke-width', 2)
                .attr('d', line);

            // Playhead
            timelineSvg.append('line')
                .attr('id', 'playhead')
                .attr('y1', 0)
                .attr('y2', height)
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);

            // X axis
            timelineSvg.append('g')
                .attr('transform', `translate(0,${{height - 5}})`)
                .call(d3.axisBottom(xScale).ticks(10).tickFormat(d => `t${{d}}`))
                .selectAll('text')
                .attr('fill', '#888')
                .attr('font-size', '10px');

            // Click to seek
            timelineSvg.on('click', function(event) {{
                const [x] = d3.pointer(event);
                const timestep = Math.round(xScale.invert(x));
                setTimestep(Math.max(0, Math.min(timestep, animationData.frames.length - 1)));
            }});
        }}

        function setupControls() {{
            document.getElementById('play-btn').onclick = togglePlay;
            document.getElementById('step-back-btn').onclick = () => setTimestep(currentTimestep - 1);
            document.getElementById('step-forward-btn').onclick = () => setTimestep(currentTimestep + 1);
            document.getElementById('reset-btn').onclick = () => setTimestep(0);

            const speedSlider = document.getElementById('speed-slider');
            speedSlider.oninput = function() {{
                speed = parseFloat(this.value);
                document.getElementById('speed-value').textContent = speed + 'x';
                if (isPlaying) {{
                    clearInterval(playInterval);
                    playInterval = setInterval(nextFrame, 1000 / speed);
                }}
            }};
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            document.getElementById('play-btn').textContent = isPlaying ? 'Pause' : 'Play';

            if (isPlaying) {{
                playInterval = setInterval(nextFrame, 1000 / speed);
            }} else {{
                clearInterval(playInterval);
            }}
        }}

        function nextFrame() {{
            if (currentTimestep < animationData.frames.length - 1) {{
                setTimestep(currentTimestep + 1);
            }} else {{
                togglePlay();
            }}
        }}

        function setTimestep(ts) {{
            if (ts < 0 || ts >= animationData.frames.length) return;
            currentTimestep = ts;
            updateFrame(ts);
        }}

        function updateFrame(timestep) {{
            const frame = animationData.frames[timestep];
            if (!frame) return;

            // Update displays
            document.getElementById('timestep-display').textContent = timestep;
            document.getElementById('price-display').textContent = (frame.market_price || 0.5).toFixed(3);
            document.getElementById('events-display').textContent = (frame.events || []).length;

            // Update playhead
            const prices = animationData.timeline.prices;
            if (prices.length > 0) {{
                const timelineWidth = document.getElementById('timeline-chart').clientWidth - 40;
                const x = 30 + (timestep / (prices.length - 1)) * (timelineWidth - 30);
                d3.select('#playhead').attr('x1', x).attr('x2', x);
            }}

            // Update agent node colors, values, and positions based on beliefs
            const beliefs = frame.agent_beliefs || {{}};
            const positions = frame.agent_positions || {{}};
            nodeGroup.selectAll('g').each(function(d) {{
                if (d.type === 'agent' && beliefs[d.id] !== undefined) {{
                    const belief = beliefs[d.id];
                    d3.select(this).select('circle')
                        .transition()
                        .duration(300)
                        .attr('fill', beliefColorScale(belief));
                    // Update text value inside node
                    d3.select(this).select('.node-value')
                        .text(belief.toFixed(2));
                    // Update position text below label (show net position)
                    const pos = positions[d.id] || {{ YES: 0, NO: 0 }};
                    const netPos = pos.YES - pos.NO;
                    const netText = netPos >= 0 ? `+${{netPos.toFixed(0)}}` : `${{netPos.toFixed(0)}}`;
                    const color = netPos > 0 ? '#4CAF50' : netPos < 0 ? '#f44336' : '#888';
                    d3.select(this).select('.node-position')
                        .attr('fill', color)
                        .text(`Net: ${{netText}}`);
                }}
            }});

            // Update market node with price
            const marketPrice = frame.market_price || 0.5;
            nodeGroup.selectAll('g').each(function(d) {{
                if (d.type === 'market') {{
                    d3.select(this).select('circle')
                        .transition()
                        .duration(300)
                        .attr('fill', beliefColorScale(marketPrice));
                    // Update text value inside market node
                    d3.select(this).select('.node-value')
                        .text(marketPrice.toFixed(2));
                }}
            }});

            // Animate events
            const events = frame.events || [];
            animateEvents(events);

            // Update event list
            updateEventList(events);
        }}

        function animateEvents(events) {{
            // Clear old particles
            particleGroup.selectAll('*').remove();

            events.forEach((event, i) => {{
                setTimeout(() => {{
                    if (event.type === 'source_emit') {{
                        animateSourceEmit(event);
                    }} else if (event.type === 'trade') {{
                        animateTrade(event);
                    }} else if (event.type === 'agent_crosspost') {{
                        animateCrosspost(event);
                    }}
                }}, i * 100);
            }});
        }}

        function animateSourceEmit(event) {{
            const sourceNode = nodes.find(n => n.id === event.source_id);
            if (!sourceNode) return;

            // Pulse source node
            nodeGroup.selectAll('g')
                .filter(d => d.id === event.source_id)
                .select('circle')
                .classed('pulsing', true)
                .on('animationend', function() {{
                    d3.select(this).classed('pulsing', false);
                }});

            // Animate particles to recipients
            (event.recipients || []).forEach((recipientId, i) => {{
                const targetNode = nodes.find(n => n.id === recipientId);
                if (!targetNode) return;

                setTimeout(() => {{
                    const particle = particleGroup.append('circle')
                        .attr('class', 'particle')
                        .attr('r', 5)
                        .attr('cx', sourceNode.x)
                        .attr('cy', sourceNode.y);

                    particle.transition()
                        .duration(500)
                        .attr('cx', targetNode.x)
                        .attr('cy', targetNode.y)
                        .on('end', function() {{
                            d3.select(this).remove();
                            // Pulse recipient
                            nodeGroup.selectAll('g')
                                .filter(d => d.id === recipientId)
                                .select('circle')
                                .classed('pulsing', true)
                                .on('animationend', function() {{
                                    d3.select(this).classed('pulsing', false);
                                }});
                        }});
                }}, i * 50);
            }});
        }}

        function animateTrade(event) {{
            const agentNode = nodes.find(n => n.id === event.agent_id);
            const marketNode = nodes.find(n => n.type === 'market');
            if (!agentNode || !marketNode) return;

            const color = event.side === 'BUY' ? '#2ECC71' : '#E74C3C';

            const particle = particleGroup.append('circle')
                .attr('class', 'particle')
                .attr('r', 8)
                .attr('fill', color)
                .attr('cx', agentNode.x)
                .attr('cy', agentNode.y);

            particle.transition()
                .duration(400)
                .attr('cx', marketNode.x)
                .attr('cy', marketNode.y)
                .attr('r', 4)
                .on('end', function() {{
                    d3.select(this).remove();
                }});
        }}

        function animateCrosspost(event) {{
            const agentNode = nodes.find(n => n.id === event.agent_id);
            const targetNode = nodes.find(n => n.id === event.target_channel);
            if (!agentNode || !targetNode) return;

            const particle = particleGroup.append('circle')
                .attr('class', 'particle')
                .attr('r', 4)
                .attr('fill', '#9B59B6')
                .attr('cx', agentNode.x)
                .attr('cy', agentNode.y);

            particle.transition()
                .duration(400)
                .attr('cx', targetNode.x)
                .attr('cy', targetNode.y)
                .on('end', function() {{
                    d3.select(this).remove();
                }});
        }}

        function updateEventList(events) {{
            const container = document.getElementById('event-list');
            container.innerHTML = '';

            if (events.length === 0) {{
                container.innerHTML = '<p style="color: #666; font-size: 0.85rem;">No events this timestep</p>';
                return;
            }}

            events.forEach(event => {{
                const div = document.createElement('div');
                div.className = 'event-item';

                const typeColors = {{
                    'source_emit': '#1DA1F2',
                    'belief_update': '#F39C12',
                    'trade': '#2ECC71',
                    'agent_crosspost': '#9B59B6',
                    'agent_receive': '#3498DB'
                }};

                div.style.borderLeftColor = typeColors[event.type] || '#e94560';

                let content = '';
                switch (event.type) {{
                    case 'source_emit':
                        content = `<strong>${{event.source_id}}</strong> emitted event to ${{event.num_recipients}} agents`;
                        break;
                    case 'belief_update':
                        const delta = (event.belief_after - event.belief_before).toFixed(3);
                        const sign = delta >= 0 ? '+' : '';
                        content = `<strong>${{event.agent_id}}</strong> belief: ${{event.belief_before.toFixed(3)}} &rarr; ${{event.belief_after.toFixed(3)}} (${{sign}}${{delta}})`;
                        break;
                    case 'trade':
                        content = `<strong>${{event.agent_id}}</strong> ${{event.side}} ${{event.shares.toFixed(1)}} shares @ ${{event.price.toFixed(3)}}`;
                        break;
                    case 'agent_crosspost':
                        content = `<strong>${{event.agent_id}}</strong> posted to <strong>${{event.target_channel}}</strong>`;
                        break;
                    default:
                        content = JSON.stringify(event);
                }}

                div.innerHTML = `
                    <div class="event-type">${{event.type.replace('_', ' ')}}</div>
                    <div class="event-content">${{content}}</div>
                `;
                container.appendChild(div);
            }});
        }}

        function showTooltip(event, d) {{
            const tooltip = document.getElementById('tooltip');
            let content = `<strong>${{d.label || d.id}}</strong><br>Type: ${{d.type}}`;

            if (d.type === 'agent') {{
                const frame = animationData.frames[currentTimestep];
                const belief = frame?.agent_beliefs?.[d.id];
                if (belief !== undefined) {{
                    content += `<br>Belief: ${{belief.toFixed(3)}}`;
                }}
                if (d.subscriptions) {{
                    content += `<br>Subscriptions: ${{d.subscriptions.join(', ')}}`;
                }}
            }}

            tooltip.innerHTML = content;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY + 10) + 'px';
        }}

        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        // Initialize on load
        window.addEventListener('load', init);
        window.addEventListener('resize', () => {{
            // Re-initialize on resize
            d3.select('#graph').selectAll('*').remove();
            setupGraph();
        }});
    </script>
</body>
</html>
'''


def render_html_animation(
    exporter: AnimationExporter,
    output_path: Optional[Path] = None,
) -> Path:
    """Render interactive HTML animation from animation data.

    Args:
        exporter: AnimationExporter with loaded data
        output_path: Output HTML file path (defaults to {run_id}_animation.html)

    Returns:
        Path to generated HTML file
    """
    if output_path is None:
        output_path = Path(f"{exporter.run_id}_animation.html")

    # Get D3-ready data
    animation_data = exporter.export_for_d3()

    # Generate HTML with embedded data
    html_content = HTML_TEMPLATE.format(
        run_id=exporter.run_id,
        animation_data=json.dumps(animation_data, indent=2)
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


def render_html_from_logs(
    log_dir: Path,
    run_id: str,
    output_path: Optional[Path] = None,
) -> Path:
    """Render HTML animation directly from log directory.

    Args:
        log_dir: Directory containing simulation logs
        run_id: Run identifier
        output_path: Output HTML file path

    Returns:
        Path to generated HTML file
    """
    exporter = AnimationExporter.from_simulation_logs(log_dir, run_id)
    return render_html_animation(exporter, output_path)
