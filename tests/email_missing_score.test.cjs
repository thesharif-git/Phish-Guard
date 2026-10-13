const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({});
vm.runInContext(fs.readFileSync('static/js/email-result-policy.js', 'utf8'), context);

const score = context.getEmailDisplayScore;
assert.equal(score({final_score:null}), null);
assert.throws(() => score({}), /invalid risk score/);
assert.equal(score({final_score:0}), 0);
assert.equal(score({final_score:82}), 82);
for (const invalid of [undefined, NaN, Infinity, -1, 101, '', '50', false, {}])
  assert.throws(() => score({final_score:invalid}), /invalid risk score/);

function element() {
  return {
    innerHTML: '<div style="width:50%"></div>', textContent:'50', className:'',
    attrs:{role:'progressbar', 'aria-valuenow':'50'}, classes:new Set(),
    setAttribute(key, value) { this.attrs[key] = value; },
    removeAttribute(key) { delete this.attrs[key]; },
    classList:{toggle(){}},
  };
}
const elements = Object.fromEntries(['confidence-bar','score-text','score-suffix'].map(id => [id, element()]));
elements['score-suffix'].classList.toggle = (name, active) => {
  if (active) elements['score-suffix'].classes.add(name);
  else elements['score-suffix'].classes.delete(name);
};
const root = {getElementById:id => elements[id]};
const unavailable = {final_score:null, security_action:'WARNING_BANNER', content_analysis:{ml_probability:null}};
assert.equal(context.getEmailDecisionStyle(unavailable).label, 'Review required');
context.updateEmailScoreDisplay(unavailable, context.getEmailDecisionStyle(unavailable), root);
assert.equal(elements['confidence-bar'].innerHTML, '');
assert.equal(elements['confidence-bar'].attrs.role, undefined);
assert.equal(elements['confidence-bar'].attrs['aria-valuenow'], undefined);
assert.equal(elements['score-text'].textContent, 'Unavailable');
assert(elements['score-suffix'].classes.has('hidden'));

for (const numeric of [0, 82]) {
  const data = {final_score:numeric, security_action:'WARNING_BANNER', content_analysis:{ml_probability:numeric / 100}};
  context.updateEmailScoreDisplay(data, context.getEmailDecisionStyle(data), root);
  assert(elements['confidence-bar'].innerHTML.includes(`width:${numeric}%`));
  assert.equal(elements['confidence-bar'].attrs['aria-valuenow'], String(numeric));
  assert.equal(elements['score-text'].textContent, String(numeric));
  assert(!elements['score-suffix'].classes.has('hidden'));
}
console.log('Email score unavailability, zero preservation, malformed-score and DOM checks passed');
