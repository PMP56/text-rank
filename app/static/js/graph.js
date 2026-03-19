const epochSlider  = document.getElementById('epochSlider');
const epochVal     = document.getElementById('epochVal');
const dampingSlider = document.getElementById('dampingSlider');
const dampingVal   = document.getElementById('dampingVal');

if (epochSlider) {
  epochSlider.addEventListener('input', () => {
    epochVal.textContent = epochSlider.value;
  });
}

if (dampingSlider) {
  dampingSlider.addEventListener('input', () => {
    dampingVal.textContent = parseFloat(dampingSlider.value).toFixed(2);
  });
}

const inputText = document.getElementById('inputText');
const charCount = document.getElementById('charCount');

function updateCount() {
  const n = inputText.value.length;
  charCount.textContent = n.toLocaleString() + ' character' + (n !== 1 ? 's' : '');
}

inputText.addEventListener('input', updateCount);
updateCount();

document.getElementById('mainForm').addEventListener('submit', function () {
  const btn = document.getElementById('submitBtn');
  btn.value = 'Analyzing…';
  btn.classList.add('loading');
});

const container = document.getElementById('graph-container');

if (container) {
  fetch('/graph-data')
    .then(r => r.json())
    .then(data => renderGraph(data))
    .catch(e => console.error('Graph load failed:', e));
}

function renderGraph({ nodes, edges }) {
  if (!nodes || nodes.length === 0) return;

  const tooltip = document.getElementById('tooltip');
  const W = container.clientWidth;
  const H = container.clientHeight;

  const palette = [
    '#c8a96e', '#6e9ecf', '#8fc89a', '#c87e6e',
    '#9e6ec8', '#c8c46e', '#6ec8c4', '#c86e9e',
    '#a0c87e', '#c8956e',
  ];

  // Map node id → array index for D3 links
  const d3nodes = nodes.map((n, i) => ({ ...n, _i: i }));
  const idxById = new Map(d3nodes.map(n => [n.id, n._i]));
  const nodeSet = new Set(nodes.map(n => n.id));

  const d3links = edges
    .filter(e => nodeSet.has(e.source) && nodeSet.has(e.target))
    .map(e => ({
      source: idxById.get(e.source),
      target: idxById.get(e.target),
      weight: e.weight,
    }));

  const svg = d3.select('#graph-container')
    .append('svg')
    .attr('width', W)
    .attr('height', H);

  const g = svg.append('g');

  svg.call(
    d3.zoom()
      .scaleExtent([0.25, 4])
      .on('zoom', e => g.attr('transform', e.transform))
  );

  const sizeScale = d3.scaleLinear().domain([0, 1]).range([13, 38]);

  const sim = d3.forceSimulation(d3nodes)
    .force('link',
      d3.forceLink(d3links)
        .id(d => d._i)
        .distance(d => 90 + (1 - d.weight) * 50)
        .strength(d => d.weight * 0.5 + 0.2)
    )
    .force('charge', d3.forceManyBody().strength(-240))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collision', d3.forceCollide().radius(d => sizeScale(d.size) + 8));

  // Edges
  const link = g.append('g')
    .selectAll('line')
    .data(d3links)
    .join('line')
    .attr('stroke', '#2c2c2c')
    .attr('stroke-width', d => Math.max(1, d.weight * 3.5))
    .attr('stroke-opacity', 0.9);

  // Node groups
  const node = g.append('g')
    .selectAll('g')
    .data(d3nodes)
    .join('g')
    .style('cursor', 'pointer')
    .call(
      d3.drag()
        .on('start', (ev, d) => {
          if (!ev.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on('end', (ev, d) => {
          if (!ev.active) sim.alphaTarget(0);
          d.fx = null; d.fy = null;
        })
    );

  node.append('circle')
    .attr('r', d => sizeScale(d.size))
    .attr('fill', (d, i) => palette[i % palette.length])
    .attr('fill-opacity', 0.15)
    .attr('stroke', (d, i) => palette[i % palette.length])
    .attr('stroke-width', 1.5);

  node.append('text')
    .text(d => d.label)
    .attr('text-anchor', 'middle')
    .attr('dominant-baseline', 'middle')
    .attr('font-family', 'JetBrains Mono, monospace')
    .attr('font-size', d => Math.max(9, sizeScale(d.size) * 0.44))
    .attr('fill', (d, i) => palette[i % palette.length])
    .attr('pointer-events', 'none');

  // Tooltip
  node
    .on('mouseenter', function (ev, d) {
      const r = container.getBoundingClientRect();
      tooltip.innerHTML = `${d.label}<br><span class="tt-rank">rank ${d.rank.toFixed(4)}</span>`;
      tooltip.classList.add('visible');
      tooltip.style.left = (ev.clientX - r.left + 14) + 'px';
      tooltip.style.top  = (ev.clientY - r.top  - 12) + 'px';
    })
    .on('mousemove', function (ev) {
      const r = container.getBoundingClientRect();
      tooltip.style.left = (ev.clientX - r.left + 14) + 'px';
      tooltip.style.top  = (ev.clientY - r.top  - 12) + 'px';
    })
    .on('mouseleave', () => tooltip.classList.remove('visible'));

  // Click to highlight neighbourhood
  node.on('click', function (ev, d) {
    ev.stopPropagation();
    const connected = new Set([d._i]);
    d3links.forEach(l => {
      if (l.source._i === d._i) connected.add(l.target._i);
      if (l.target._i === d._i) connected.add(l.source._i);
    });
    node.selectAll('circle')
      .attr('fill-opacity', n => connected.has(n._i) ? 0.35 : 0.05)
      .attr('stroke-opacity', n => connected.has(n._i) ? 1 : 0.2);
    node.selectAll('text')
      .attr('opacity', n => connected.has(n._i) ? 1 : 0.2);
    link.attr('stroke-opacity',
      l => (l.source._i === d._i || l.target._i === d._i) ? 0.85 : 0.06);
  });

  // Background click resets highlight
  svg.on('click', () => {
    node.selectAll('circle').attr('fill-opacity', 0.15).attr('stroke-opacity', 1);
    node.selectAll('text').attr('opacity', 1);
    link.attr('stroke-opacity', 0.9);
  });

  sim.on('tick', () => {
    link
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });
}