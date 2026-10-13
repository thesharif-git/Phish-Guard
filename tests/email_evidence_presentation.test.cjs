const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({});
vm.runInContext(fs.readFileSync('static/js/email-result-policy.js', 'utf8'), context);
const describe = context.getEvidenceMismatchSummary;
const actual = {type:'VISIBLE_LINK_HOST_DIFFERENCE', evidence:'Displayed host differs from href host; delegation is unverified.'};
assert.equal(describe([actual]), actual.evidence);
assert.equal(describe([actual, {...actual}]), actual.evidence);
assert.equal(describe(['OLDER_SAVED_SIGNAL', {type:'OTHER_RELATION'}]), 'OLDER SAVED SIGNAL OTHER RELATION');
assert.equal(describe([null, {}, 42, false]), '');
assert.equal(describe(undefined), '');
assert.equal(describe({mismatches:[actual]}), '');
// Both consumers use the object-aware shared component, not String.replace on objects.
for (const path of ['templates/scan.html', 'templates/url_scan.html']) {
  const template = fs.readFileSync(path, 'utf8');
  assert.ok(template.includes('getEvidenceMismatchSummary('));
  assert.ok(template.includes('/static/js/email-result-policy.js'));
  assert.equal(/m\.replace\(/.test(template), false);
}
const template = fs.readFileSync('templates/scan.html', 'utf8');
const source = template.slice(template.indexOf('function createRiskBar'), template.indexOf('function createFindingRow'));
const bars = vm.createContext({escapeHtml:value=>String(value)});
vm.runInContext(source, bars);
assert.match(bars.createRiskBar('Content review cues', 90), /Unvalidated/);
assert.doesNotMatch(bars.createRiskBar('Content review cues', 90), /High risk|Suspicious|Low risk/);
assert.match(bars.createRiskBar('Email model output', 2, true), /2\/100/);
assert.match(bars.createRiskBar('Unavailable identity', null), /Unavailable/);
console.log('Object evidence rendering, legacy records and unvalidated review bars passed');
