// ======================================================
//  BMI Kiosk — App Logic
//  SVG Gauge · Skeleton Overlay · Socket.IO
// ======================================================

const socket = io();

// ── State ───────────────────────────────────────────
let isScanning = false;
let currentSex = 'male';
let scanCount = 0;
let skeletonAnimFrame = null;

// ── DOM Elements ────────────────────────────────────
const els = {
    btnMale:      document.getElementById('btn-male'),
    btnFemale:    document.getElementById('btn-female'),
    ageInput:     document.getElementById('age-input'),
    btnScan:      document.getElementById('btn-scan'),
    statusBadge:  document.getElementById('status-badge'),
    focusBox:     document.getElementById('focus-box'),
    videoFeed:    document.getElementById('video-feed'),
    historyTbody: document.getElementById('history-tbody'),
    emptyHistory: document.getElementById('empty-history'),
    historyCount: document.getElementById('history-count'),
    valHeight:    document.getElementById('val-height'),
    valWeight:    document.getElementById('val-weight'),
    valBmi:       document.getElementById('val-bmi'),
    valBfp:       document.getElementById('val-bfp'),
    gaugeValue:   document.getElementById('gauge-value'),
    gaugeLabel:   document.getElementById('gauge-label'),
    gaugeNeedle:  document.getElementById('gauge-needle'),
    gaugeBg:      document.getElementById('gauge-bg'),
    gaugeFill:    document.getElementById('gauge-fill'),
    gaugeTicks:   document.getElementById('gauge-ticks'),
    skeletonBones:  document.getElementById('skeleton-bones'),
    skeletonJoints: document.getElementById('skeleton-joints'),
    skeletonBbox:   document.getElementById('skeleton-bbox'),
};


// ═══════════════════════════════════════════════════
//  SVG SEMI-CIRCLE GAUGE
// ═══════════════════════════════════════════════════

const GAUGE = {
    cx: 100,
    cy: 100,
    r: 78,
    min: 12,
    max: 40,
    currentAngle: Math.PI,
};

function arcPath(cx, cy, r, startAngle, endAngle) {
    // Angles: π = left, 0 = right (standard math, Y-up)
    const x1 = cx + r * Math.cos(startAngle);
    const y1 = cy - r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle);
    const y2 = cy - r * Math.sin(endAngle);
    const sweep = (startAngle - endAngle) > Math.PI ? 1 : 0;
    return `M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${r} ${r} 0 ${sweep} 0 ${x2.toFixed(1)} ${y2.toFixed(1)}`;
}

function initGauge() {
    const { cx, cy, r } = GAUGE;
    const path = arcPath(cx, cy, r, Math.PI, 0);
    els.gaugeBg.setAttribute('d', path);
    els.gaugeFill.setAttribute('d', path);

    // Tick labels
    const labels = [
        { val: 18.5, text: '18.5' },
        { val: 25,   text: '25' },
        { val: 30,   text: '30' },
    ];
    els.gaugeTicks.innerHTML = '';
    labels.forEach(({ val, text }) => {
        const frac = (val - GAUGE.min) / (GAUGE.max - GAUGE.min);
        const angle = Math.PI - frac * Math.PI;
        const labelR = r + 18;
        const lx = cx + labelR * Math.cos(angle);
        const ly = cy - labelR * Math.sin(angle);
        const el = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        el.setAttribute('x', lx);
        el.setAttribute('y', ly);
        el.setAttribute('text-anchor', 'middle');
        el.setAttribute('dominant-baseline', 'middle');
        el.setAttribute('fill', '#4b5563');
        el.setAttribute('font-size', '8');
        el.textContent = text;
        els.gaugeTicks.appendChild(el);

        // Small tick line
        const t1R = r - 3;
        const t2R = r + 5;
        const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        tick.setAttribute('x1', cx + t1R * Math.cos(angle));
        tick.setAttribute('y1', cy - t1R * Math.sin(angle));
        tick.setAttribute('x2', cx + t2R * Math.cos(angle));
        tick.setAttribute('y2', cy - t2R * Math.sin(angle));
        tick.setAttribute('stroke', '#374151');
        tick.setAttribute('stroke-width', '1');
        els.gaugeTicks.appendChild(tick);
    });

    // Default needle position (left)
    setNeedleAngle(Math.PI);
}

function setNeedleAngle(angle) {
    const { cx, cy, r } = GAUGE;
    const len = r - 14;
    const nx = cx + len * Math.cos(angle);
    const ny = cy - len * Math.sin(angle);
    els.gaugeNeedle.setAttribute('x2', nx.toFixed(1));
    els.gaugeNeedle.setAttribute('y2', ny.toFixed(1));
}

function updateGauge(bmi, category) {
    const clamped = Math.max(GAUGE.min, Math.min(GAUGE.max, bmi));
    const frac = (clamped - GAUGE.min) / (GAUGE.max - GAUGE.min);
    const targetAngle = Math.PI - frac * Math.PI;

    // Animate needle
    const start = GAUGE.currentAngle;
    const t0 = performance.now();
    const dur = 1200;

    function step(now) {
        const t = Math.min((now - t0) / dur, 1);
        const ease = 1 - Math.pow(1 - t, 3);
        setNeedleAngle(start + (targetAngle - start) * ease);
        if (t < 1) requestAnimationFrame(step);
        else GAUGE.currentAngle = targetAngle;
    }
    requestAnimationFrame(step);

    // Center text
    els.gaugeValue.textContent = bmi.toFixed(1);

    const colors = { Lean: '#38bdf8', Healthy: '#4ade80', Overweight: '#fbbf24', Obese: '#f43f5e' };
    els.gaugeLabel.textContent = category;
    els.gaugeLabel.style.color = colors[category] || '#64748b';
}

function resetGauge() {
    GAUGE.currentAngle = Math.PI;
    setNeedleAngle(Math.PI);
    els.gaugeValue.textContent = '--';
    els.gaugeLabel.textContent = 'Pending';
    els.gaugeLabel.style.color = '';
}


// ═══════════════════════════════════════════════════
//  SKELETON OVERLAY (Subtle mock / real data ready)
// ═══════════════════════════════════════════════════

// COCO 17 keypoint connections
const BONE_PAIRS = [
    [0,1],[0,2],[1,3],[2,4],       // head
    [5,6],                          // shoulders
    [5,7],[7,9],                    // left arm
    [6,8],[8,10],                   // right arm
    [5,11],[6,12],                  // torso
    [11,12],                        // hips
    [11,13],[13,15],                // left leg
    [12,14],[14,16],                // right leg
];
const MAJOR_JOINTS = new Set([5,6,11,12,13,14]);

// Normalized base pose (0–1)
const BASE_POSE = [
    [0.50,0.13],[0.48,0.11],[0.52,0.11],[0.45,0.13],[0.55,0.13],  // head
    [0.41,0.26],[0.59,0.26],                                        // shoulders
    [0.36,0.40],[0.64,0.40],                                        // elbows
    [0.34,0.52],[0.66,0.52],                                        // wrists
    [0.44,0.52],[0.56,0.52],                                        // hips
    [0.43,0.68],[0.57,0.68],                                        // knees
    [0.42,0.84],[0.58,0.84],                                        // ankles
];

const SK_W = 640, SK_H = 480;

function initSkeleton() {
    els.skeletonBones.innerHTML = '';
    els.skeletonJoints.innerHTML = '';

    BONE_PAIRS.forEach((_, i) => {
        const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        ln.id = `bone-${i}`;
        ln.classList.add('skeleton-bone');
        els.skeletonBones.appendChild(ln);
    });

    BASE_POSE.forEach((_, i) => {
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.id = `joint-${i}`;
        const isMajor = MAJOR_JOINTS.has(i);
        c.setAttribute('r', isMajor ? '4' : '2.5');
        c.classList.add(isMajor ? 'skeleton-joint-major' : 'skeleton-joint');
        if (isMajor) c.setAttribute('filter', 'url(#glow-joint)');
        els.skeletonJoints.appendChild(c);
    });

    animateIdleSkeleton();
}

function animateIdleSkeleton() {
    const t0 = performance.now();

    function frame(now) {
        const t = (now - t0) / 1000;
        const pose = BASE_POSE.map(([bx, by], i) => {
            let dx = 0, dy = 0;
            if (i >= 5 && i <= 10)  dy = Math.sin(t * 1.3) * 0.003;           // breathing
            if (i >= 11 && i <= 14) dx = Math.sin(t * 0.7 + i * 0.4) * 0.002; // sway
            if (i === 9 || i === 10) { dx = Math.sin(t * 1.1 + i) * 0.004; dy = Math.cos(t * 0.8 + i) * 0.003; }
            if (i <= 4) dy = Math.sin(t * 0.6) * 0.002;
            return [bx + dx, by + dy];
        });
        renderSkeleton(pose);
        skeletonAnimFrame = requestAnimationFrame(frame);
    }
    skeletonAnimFrame = requestAnimationFrame(frame);
}

function renderSkeleton(pose) {
    BONE_PAIRS.forEach(([a, b], i) => {
        const ln = document.getElementById(`bone-${i}`);
        if (!ln) return;
        ln.setAttribute('x1', (pose[a][0] * SK_W).toFixed(1));
        ln.setAttribute('y1', (pose[a][1] * SK_H).toFixed(1));
        ln.setAttribute('x2', (pose[b][0] * SK_W).toFixed(1));
        ln.setAttribute('y2', (pose[b][1] * SK_H).toFixed(1));
    });
    pose.forEach(([x, y], i) => {
        const c = document.getElementById(`joint-${i}`);
        if (!c) return;
        c.setAttribute('cx', (x * SK_W).toFixed(1));
        c.setAttribute('cy', (y * SK_H).toFixed(1));
    });
    // Bounding box
    const xs = pose.map(p => p[0] * SK_W);
    const ys = pose.map(p => p[1] * SK_H);
    const pad = 24;
    const bx = Math.min(...xs) - pad, by = Math.min(...ys) - pad;
    els.skeletonBbox.setAttribute('x', bx.toFixed(1));
    els.skeletonBbox.setAttribute('y', by.toFixed(1));
    els.skeletonBbox.setAttribute('width', (Math.max(...xs) + pad - bx).toFixed(1));
    els.skeletonBbox.setAttribute('height', (Math.max(...ys) + pad - by).toFixed(1));
}

// Accept real landmark data from backend
function updateSkeletonFromBackend(landmarks) {
    if (skeletonAnimFrame) { cancelAnimationFrame(skeletonAnimFrame); skeletonAnimFrame = null; }
    renderSkeleton(landmarks.map(lm => [lm.x, lm.y]));
}


// ═══════════════════════════════════════════════════
//  SEX TOGGLE
// ═══════════════════════════════════════════════════

function setSex(sex) {
    currentSex = sex;
    els.btnMale.classList.toggle('active', sex === 'male');
    els.btnFemale.classList.toggle('active', sex === 'female');
}


// ═══════════════════════════════════════════════════
//  SCAN / RESET
// ═══════════════════════════════════════════════════

function triggerScan() {
    if (isScanning) return;
    socket.emit('trigger_scan', { age: parseInt(els.ageInput.value) || 25, sex: currentSex });
}
function resetUI() { socket.emit('reset'); }

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); triggerScan(); }
    else if (e.code === 'KeyR') resetUI();
});


// ═══════════════════════════════════════════════════
//  SOCKET.IO EVENTS
// ═══════════════════════════════════════════════════

socket.on('scan_started', () => {
    isScanning = true;
    els.btnScan.disabled = true;
    els.btnScan.querySelector('.btn-text').innerText = "ANALYZING\u2026";
    const ico = els.btnScan.querySelector('.btn-icon');
    ico.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>';
    ico.classList.add('animate-spin');

    els.statusBadge.className = 'state-scanning';
    Object.assign(els.statusBadge.style, { background:'rgba(94,234,212,0.06)', border:'1px solid rgba(94,234,212,0.25)', color:'#5eead4' });
    els.statusBadge.querySelector('.status-text').innerText = "Analyzing";
    els.statusBadge.querySelector('.status-dot').style.background = '#5eead4';

    els.focusBox.style.borderColor = 'transparent';
    els.focusBox.classList.add('scanning');
    els.videoFeed.style.opacity = '0.8';
});

socket.on('scan_result', (data) => {
    isScanning = false;
    els.btnScan.disabled = false;
    els.btnScan.querySelector('.btn-text').innerText = "BEGIN ANALYSIS";
    const ico = els.btnScan.querySelector('.btn-icon');
    ico.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3"/>';
    ico.classList.remove('animate-spin');

    els.statusBadge.className = 'state-complete';
    Object.assign(els.statusBadge.style, { background:'rgba(52,211,153,0.06)', border:'1px solid rgba(52,211,153,0.25)', color:'#34d399' });
    els.statusBadge.querySelector('.status-text').innerText = "Complete";
    els.statusBadge.querySelector('.status-dot').style.background = '#34d399';

    els.focusBox.classList.remove('scanning');
    els.focusBox.classList.add('complete');
    els.videoFeed.style.opacity = '1';

    animateValue(els.valHeight, parseFloat(els.valHeight.innerText) || 0, data.height_m, 2);
    animateValue(els.valWeight, parseFloat(els.valWeight.innerText) || 0, data.weight_kg, 1);
    animateValue(els.valBmi,    parseFloat(els.valBmi.innerText)    || 0, data.bmi, 1);
    animateValue(els.valBfp,    parseFloat(els.valBfp.innerText)    || 0, data.body_fat_pct, 1);

    updateGauge(data.bmi, data.category);
    addHistoryRow(data);

    setTimeout(() => {
        if (!isScanning) {
            els.statusBadge.className = 'state-ready';
            Object.assign(els.statusBadge.style, { background:'rgba(19,21,26,0.9)', border:'1px solid rgba(255,255,255,0.08)', color:'#94a3b8' });
            els.statusBadge.querySelector('.status-text').innerText = "Ready";
            els.statusBadge.querySelector('.status-dot').style.background = '#34d399';
            els.focusBox.classList.remove('complete');
            els.focusBox.style.borderColor = 'rgba(94,234,212,0.2)';
        }
    }, 3000);
});

socket.on('ui_reset', () => {
    els.valHeight.innerText = '--';
    els.valWeight.innerText = '--';
    els.valBmi.innerText = '--';
    els.valBfp.innerText = '--';
    resetGauge();
    els.statusBadge.className = 'state-ready';
    Object.assign(els.statusBadge.style, { background:'rgba(19,21,26,0.9)', border:'1px solid rgba(255,255,255,0.08)', color:'#94a3b8' });
    els.statusBadge.querySelector('.status-text').innerText = "Ready";
    els.statusBadge.querySelector('.status-dot').style.background = '#34d399';
    els.focusBox.classList.remove('scanning', 'complete');
    els.focusBox.style.borderColor = 'rgba(94,234,212,0.2)';
});

socket.on('pose_landmarks', (data) => {
    if (data && data.landmarks) updateSkeletonFromBackend(data.landmarks);
});


// ═══════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════

function animateValue(obj, start, end, decimals) {
    let t0 = null;
    const dur = 1000;
    obj.classList.add('updating');
    function step(ts) {
        if (!t0) t0 = ts;
        const p = Math.min((ts - t0) / dur, 1);
        const ease = 1 - Math.pow(1 - p, 4);
        obj.innerHTML = (start + (end - start) * ease).toFixed(decimals);
        if (p < 1) requestAnimationFrame(step);
        else obj.classList.remove('updating');
    }
    requestAnimationFrame(step);
}

function addHistoryRow(data) {
    if (scanCount === 0) els.emptyHistory.style.display = 'none';
    scanCount++;
    els.historyCount.innerText = `${scanCount} Record${scanCount !== 1 ? 's' : ''}`;

    const time = new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' });

    const catColors = {
        Lean:       { bg:'rgba(56,189,248,0.1)',  text:'#38bdf8', border:'rgba(56,189,248,0.2)' },
        Healthy:    { bg:'rgba(74,222,128,0.1)',   text:'#4ade80', border:'rgba(74,222,128,0.2)' },
        Overweight: { bg:'rgba(251,191,36,0.1)',   text:'#fbbf24', border:'rgba(251,191,36,0.2)' },
        Obese:      { bg:'rgba(244,63,94,0.1)',    text:'#f43f5e', border:'rgba(244,63,94,0.2)' },
    };
    const cc = catColors[data.category] || catColors.Obese;

    const tr = document.createElement('tr');
    tr.className = 'new-row';
    tr.style.borderBottom = '1px solid rgba(255,255,255,0.04)';
    tr.innerHTML = `
        <td style="padding:12px 24px; color:#4b5563; font-size:12px; white-space:nowrap;">${time}</td>
        <td style="padding:12px 24px; font-size:13px;">${data.height_m.toFixed(2)}<span style="font-size:10px;color:#4b5563;margin-left:2px;">m</span></td>
        <td style="padding:12px 24px; font-size:13px; font-weight:600; color:#fff;">${data.weight_kg.toFixed(1)}<span style="font-size:10px;color:#4b5563;margin-left:2px;">kg</span></td>
        <td style="padding:12px 24px; font-size:13px; color:#5eead4; font-weight:600;">${data.bmi.toFixed(1)}</td>
        <td style="padding:12px 24px; font-size:13px;">${data.body_fat_pct.toFixed(1)}<span style="font-size:10px;color:#4b5563;margin-left:2px;">%</span></td>
        <td style="padding:12px 24px; font-size:13px; color:#64748b;"><span style="color:#fff;">${data.fat_mass.toFixed(1)}</span> / ${data.lean_mass.toFixed(1)}</td>
        <td style="padding:12px 24px;">
            <span style="padding:4px 12px; font-size:11px; font-weight:500; border-radius:20px; background:${cc.bg}; color:${cc.text}; border:1px solid ${cc.border};">${data.category}</span>
        </td>
    `;
    els.historyTbody.insertBefore(tr, els.historyTbody.firstChild);
    if (els.historyTbody.children.length > 10) els.historyTbody.removeChild(els.historyTbody.lastChild);
}


// ═══════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════

initGauge();
initSkeleton();
