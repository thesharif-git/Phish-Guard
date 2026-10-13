/* Presentation follows the engine decision, never a numeric score shortcut. */
function getEmailDisplayScore(data) {
  if (!data || !Object.prototype.hasOwnProperty.call(data, 'final_score'))
    throw new Error('The scan returned an invalid risk score. Please try again.');
  const score = data.final_score;
  if (score === null) return null;
  if (typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 100)
    throw new Error('The scan returned an invalid risk score. Please try again.');
  return score;
}

function updateEmailScoreDisplay(data, style, root = document) {
  const score = getEmailDisplayScore(data);
  const probability = data.content_analysis?.ml_probability;
  const hasProbability = typeof probability === 'number' && Number.isFinite(probability);
  const available = score !== null && (hasProbability || data.security_action === 'BLOCK_BY_POLICY');
  const bar = root.getElementById('confidence-bar');
  if (bar) {
    if (available) {
      bar.innerHTML = `<div class="h-full rounded-full ${style.colorClass}" style="width:${score}%"></div>`;
      bar.setAttribute('role', 'progressbar');
      bar.setAttribute('aria-label', 'Risk score');
      bar.setAttribute('aria-valuenow', String(score));
    } else {
      bar.innerHTML = '';
      bar.removeAttribute('role');
      bar.removeAttribute('aria-valuenow');
      bar.setAttribute('aria-label', 'Risk score unavailable');
    }
  }
  const value = root.getElementById('score-text');
  if (value) {
    value.textContent = available ? String(score) : 'Unavailable';
    value.className = style.textClass;
  }
  root.getElementById('score-suffix')?.classList.toggle('hidden', !available);
}

function getEmailDecisionStyle(data) {
  const action = data.security_action;
  if (action === 'BLOCK_BY_POLICY') return {colorClass:'bg-rose-500', textClass:'text-rose-400', label:'Blocked by policy', icon:'gpp_bad'};
  if (action === 'AUTO_QUARANTINE') return {colorClass:'bg-rose-500', textClass:'text-rose-400', label:'Quarantine recommended', icon:'gpp_bad'};
  if (action === 'SAFE_DELIVERY' && data.conformal_guarantee?.calibration_valid === true)
    return {colorClass:'bg-emerald-500', textClass:'text-emerald-400', label:'Low risk in completed checks', icon:'check_circle'};
  return {colorClass:'bg-amber-500', textClass:'text-amber-400', label:'Review required', icon:'warning'};
}

function getEmailAttachmentSummary(data) {
  if (data.email_parsing?.unscanned_parts?.length || data.vision_intelligence?.details?.some(d => d.qr_scan_complete === false) ||
      data.sensor_errors?.some(e => /image|vision|qr|attachment/i.test(e)))
    return {label:'Incomplete inspection', className:'text-amber-400'};
  if (data.vision_intelligence?.has_quishing) return {label:'QR payload detected', className:'text-amber-400'};
  if (data.vision_intelligence?.has_images) return {label:`${data.vision_intelligence.images_analyzed} image(s) checked for QR`, className:'text-subtext'};
  return {label:'No attachment inspected', className:'text-subtext'};
}

/* Current engines return evidence objects; older saved scans may use strings. */
function getEvidenceMismatchSummary(mismatches) {
  if (!Array.isArray(mismatches)) return '';
  const seen = new Set();
  const descriptions = [];
  for (const mismatch of mismatches) {
    let text = '';
    if (typeof mismatch === 'string') text = mismatch.replace(/_/g, ' ').trim();
    else if (mismatch && typeof mismatch === 'object') {
      if (typeof mismatch.evidence === 'string') text = mismatch.evidence.trim();
      else if (typeof mismatch.type === 'string') text = mismatch.type.replace(/_/g, ' ').trim();
    }
    const key = text.toLowerCase();
    if (text && !seen.has(key)) {
      seen.add(key);
      descriptions.push(text);
    }
  }
  return descriptions.join(' ');
}
