/* Small dependency-free SVG charts. Colours come from CSS variables so the
   charts follow the page's light/dark theme. */

const Charts = (() => {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const tooltipEl = () => document.getElementById('tooltip');

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, (ch) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
    ));
  }

  function showTip(html, clientX, clientY) {
    const tip = tooltipEl();
    tip.innerHTML = html;
    tip.hidden = false;
    const box = tip.getBoundingClientRect();
    let x = clientX + 14;
    let y = clientY + 14;
    if (x + box.width > window.innerWidth - 8) x = clientX - box.width - 14;
    if (y + box.height > window.innerHeight - 8) y = clientY - box.height - 14;
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.max(8, y)}px`;
  }

  function hideTip() {
    tooltipEl().hidden = true;
  }

  function niceTicks(min, max, count = 4) {
    if (max <= min) max = min + 1;
    const rawStep = (max - min) / count;
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rawStep);
    const start = Math.floor(min / step) * step;
    const ticks = [];
    for (let value = start; value < max + step * 0.999; value += step) ticks.push(Number(value.toFixed(6)));
    return ticks;
  }

  function el(name, attrs = {}, parent) {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (parent) parent.appendChild(node);
    return node;
  }

  function text(parent, x, y, content, attrs = {}) {
    const node = el('text', { x, y, ...attrs }, parent);
    node.textContent = content;
    return node;
  }

  function mount(container, height) {
    container.innerHTML = '';
    const width = Math.max(container.clientWidth, 280);
    const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, height, role: 'img' }, container);
    return { svg, width };
  }

  /* Line chart over a shared category axis.
     series: [{ name, color, values: (number|null)[], width?, dashed?, markers?: number[], ring?: boolean,
                endLabel?: string }] */
  function lineChart(container, { labels, series, formatY, formatTip, height = 260, yMin = null, ariaLabel = '' }) {
    const { svg, width } = mount(container, height);
    if (ariaLabel) svg.setAttribute('aria-label', ariaLabel);
    const margin = { top: 14, right: 18, bottom: 26, left: 56 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const all = series.flatMap((s) => s.values.filter((v) => v != null));
    const ticks = niceTicks(yMin ?? Math.min(...all), Math.max(...all));
    const lo = ticks[0];
    const hi = ticks[ticks.length - 1];
    const xAt = (i) => margin.left + (labels.length === 1 ? innerW / 2 : (i / (labels.length - 1)) * innerW);
    const yAt = (v) => margin.top + innerH - ((v - lo) / (hi - lo)) * innerH;

    ticks.forEach((tick) => {
      el('line', { x1: margin.left, x2: width - margin.right, y1: yAt(tick), y2: yAt(tick),
        style: `stroke:var(${tick === lo ? '--axis' : '--grid'})`, 'stroke-width': 1 }, svg);
      text(svg, margin.left - 8, yAt(tick) + 4, formatY(tick), { 'text-anchor': 'end' });
    });

    const every = Math.ceil(labels.length / Math.max(Math.floor(innerW / 64), 1));
    labels.forEach((label, i) => {
      if (i % every === 0 || i === labels.length - 1) {
        if (i !== labels.length - 1 && labels.length - 1 - i < every * 0.6) return;
        text(svg, xAt(i), height - 8, label, { 'text-anchor': i === labels.length - 1 ? 'end' : 'middle' });
      }
    });

    series.forEach((s) => {
      let path = '';
      let pen = false;
      s.values.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        path += `${pen ? 'L' : 'M'}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`;
        pen = true;
      });
      const attrs = { d: path, fill: 'none', style: `stroke:var(${s.color})`, 'stroke-width': s.width || 2,
        'stroke-linejoin': 'round', 'stroke-linecap': 'round' };
      if (s.dashed) attrs['stroke-dasharray'] = '5 5';
      el('path', attrs, svg);
      (s.markers || []).forEach((i) => {
        if (s.values[i] == null) return;
        el('circle', { cx: xAt(i), cy: yAt(s.values[i]), r: 5,
          style: s.ring
            ? `fill:var(--surface);stroke:var(${s.color});stroke-width:2.5`
            : `fill:var(${s.color});stroke:var(--surface);stroke-width:2` }, svg);
      });
    });

    const cross = el('line', { y1: margin.top, y2: margin.top + innerH, style: 'stroke:var(--axis)',
      'stroke-width': 1, visibility: 'hidden' }, svg);
    const hit = el('rect', { x: margin.left, y: margin.top, width: innerW, height: innerH, fill: 'transparent' }, svg);
    const onMove = (event) => {
      const box = svg.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * width;
      const index = Math.min(labels.length - 1, Math.max(0, Math.round(((px - margin.left) / innerW) * (labels.length - 1))));
      cross.setAttribute('x1', xAt(index));
      cross.setAttribute('x2', xAt(index));
      cross.setAttribute('visibility', 'visible');
      showTip(formatTip(index), event.clientX, event.clientY);
    };
    hit.addEventListener('pointermove', onMove);
    hit.addEventListener('pointerdown', onMove);
    hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
  }

  /* Horizontal bars with the value at the tip and an optional reference tick.
     rows: [{ label, value, ref?, tip }] */
  function barChart(container, { rows, color = '--series-1', format, labelWidth = 124, rowHeight = 28, ariaLabel = '' }) {
    const height = rows.length * rowHeight + 6;
    const { svg, width } = mount(container, height);
    labelWidth = Math.min(labelWidth, width * 0.36);
    if (ariaLabel) svg.setAttribute('aria-label', ariaLabel);
    const valueRoom = 78;
    const innerW = Math.max(width - labelWidth - valueRoom, 40);
    const max = Math.max(...rows.map((r) => Math.max(r.value, r.ref || 0)), 1);
    const xAt = (v) => labelWidth + (v / max) * innerW;
    const thickness = Math.min(16, rowHeight - 10);

    el('line', { x1: labelWidth, x2: labelWidth, y1: 0, y2: height, style: 'stroke:var(--axis)', 'stroke-width': 1 }, svg);

    rows.forEach((row, i) => {
      const cy = i * rowHeight + rowHeight / 2 + 3;
      text(svg, labelWidth - 10, cy + 4, row.label, { 'text-anchor': 'end', class: 'cat-label' });
      const w = Math.max(xAt(row.value) - labelWidth, row.value > 0 ? 2 : 0);
      const r = Math.min(4, w);
      const top = cy - thickness / 2;
      // square at the baseline, rounded at the data end
      el('path', {
        d: `M${labelWidth},${top}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${thickness - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`,
        style: `fill:var(${color})`,
      }, svg);
      if (row.ref != null) {
        el('line', { x1: xAt(row.ref), x2: xAt(row.ref), y1: top - 4, y2: top + thickness + 4,
          style: 'stroke:var(--ink)', 'stroke-width': 2 }, svg);
      }
      text(svg, width - 2, cy + 4, format(row.value), { class: 'value-label', 'text-anchor': 'end' });

      const hit = el('rect', { x: 0, y: i * rowHeight + 3, width, height: rowHeight, fill: 'transparent' }, svg);
      hit.addEventListener('pointermove', (event) => showTip(row.tip, event.clientX, event.clientY));
      hit.addEventListener('pointerleave', hideTip);
    });
  }

  /* Bars diverging from a zero line: positive in one hue, negative in the opposite.
     rows: [{ label, value, tip }] */
  function divergingChart(container, { rows, format, labelWidth = 170, rowHeight = 28, ariaLabel = '' }) {
    const height = rows.length * rowHeight + 6;
    const { svg, width } = mount(container, height);
    if (ariaLabel) svg.setAttribute('aria-label', ariaLabel);
    labelWidth = Math.min(labelWidth, width * 0.48);
    const valueRoom = width < 420 ? 40 : 58;
    const innerW = Math.max(width - labelWidth - 2 * valueRoom, 60);
    const max = Math.max(...rows.map((r) => Math.abs(r.value)), 0.01);
    const hasNegative = rows.some((r) => r.value < 0);
    const hasPositive = rows.some((r) => r.value > 0);
    const zeroX = labelWidth + valueRoom + (hasNegative ? (hasPositive ? innerW / 2 : innerW) : 0);
    const scale = (hasNegative && hasPositive ? innerW / 2 : innerW) / max;
    const thickness = 14;

    el('line', { x1: zeroX, x2: zeroX, y1: 0, y2: height, style: 'stroke:var(--axis)', 'stroke-width': 1 }, svg);

    rows.forEach((row, i) => {
      const cy = i * rowHeight + rowHeight / 2 + 3;
      text(svg, labelWidth - 10, cy + 4, row.label, { 'text-anchor': 'end', class: 'cat-label' });
      const w = Math.max(Math.abs(row.value) * scale, Math.abs(row.value) > 0 ? 2 : 0);
      const r = Math.min(4, w);
      const top = cy - thickness / 2;
      const positive = row.value >= 0;
      const d = positive
        ? `M${zeroX},${top}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${thickness - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`
        : `M${zeroX},${top}h${-(w - r)}a${r},${r} 0 0 0 ${-r},${r}v${thickness - 2 * r}a${r},${r} 0 0 0 ${r},${r}h${w - r}z`;
      el('path', { d, style: `fill:var(${positive ? '--series-1' : '--negative'})` }, svg);
      text(svg, positive ? zeroX + w + 6 : zeroX - w - 6, cy + 4, format(row.value),
        { class: 'value-label', 'text-anchor': positive ? 'start' : 'end' });

      const hit = el('rect', { x: 0, y: i * rowHeight + 3, width, height: rowHeight, fill: 'transparent' }, svg);
      hit.addEventListener('pointermove', (event) => showTip(row.tip, event.clientX, event.clientY));
      hit.addEventListener('pointerleave', hideTip);
    });
  }

  return { esc, lineChart, barChart, divergingChart, showTip, hideTip };
})();
