(() => {
  const demo = document.querySelector('[data-evidence-demo]');
  if (!demo) return;
  const descriptions = {
    sender: ['SENDER IDENTITY', 'The sender uses “0” in place of “o” in Microsoft. The displayed brand and sending domain do not match.'],
    urgency: ['MESSAGE INTENT', '“Your account will be suspended today” creates pressure to act. This language is a warning signal to consider alongside the sender and link.'],
    link: ['LINK DESTINATION', 'The button points to account-verify.example/login. Check the actual destination against the service’s known official address before opening it.']
  };
  function selectEvidence(key) {
    if (!descriptions[key]) return;
    demo.querySelectorAll('.pg-indicator-list button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.selectEvidence === key)));
    demo.querySelectorAll('[data-evidence-target]').forEach(target => target.classList.toggle('is-selected', target.dataset.evidenceTarget === key));
    demo.querySelector('.pg-explanation-label').textContent = descriptions[key][0];
    demo.querySelector('.pg-explanation-copy').textContent = descriptions[key][1];
    demo.dataset.selectedEvidence = key;
  }
  demo.querySelectorAll('[data-select-evidence]').forEach(button => button.addEventListener('click', () => selectEvidence(button.dataset.selectEvidence)));
  selectEvidence('sender');

  // Input-driven animation only: no idle loop, touch tilt, or scroll interception.
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const precisePointer = matchMedia('(hover: hover) and (pointer: fine) and (min-width: 1200px)');
  const hero = demo.closest('.pg-hero');
  let enabled = false, visible = true, frameId = 0, lastTime = 0;
  let pointerX = 0, pointerY = 0, currentX = 1, currentY = -1.5, currentDepth = 38;
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  function draw(time) {
    frameId = 0;
    if (!enabled || !visible || document.hidden) return;
    const dt = Math.min((time - lastTime) / 1000 || .016, .05); lastTime = time;
    const bounds = hero.getBoundingClientRect();
    const depthRatio = 1 - clamp(-bounds.top / (bounds.height * .65), 0, 1);
    const desiredX = (1 - pointerY * 1.5) * depthRatio;
    const desiredY = (-1.5 + pointerX) * depthRatio;
    const desiredDepth = 16 + 22 * depthRatio;
    const blend = 1 - Math.exp(-dt * 12);
    currentX += (desiredX - currentX) * blend;
    currentY += (desiredY - currentY) * blend;
    currentDepth += (desiredDepth - currentDepth) * blend;
    demo.style.setProperty('--pg-rx', `${currentX.toFixed(3)}deg`);
    demo.style.setProperty('--pg-ry', `${currentY.toFixed(3)}deg`);
    demo.style.setProperty('--pg-front', `${currentDepth.toFixed(2)}px`);
    demo.style.setProperty('--pg-evidence', `${(currentDepth * .32).toFixed(2)}px`);
    demo.style.setProperty('--pg-shift', `${(pointerY * -2 * depthRatio).toFixed(2)}px`);
    if (Math.abs(currentX - desiredX) + Math.abs(currentY - desiredY) + Math.abs(currentDepth - desiredDepth) > .015) requestDraw();
  }
  function requestDraw() { if (enabled && visible && !document.hidden && !frameId) frameId = requestAnimationFrame(draw); }
  function updateMotion() {
    enabled = !reducedMotion.matches && precisePointer.matches;
    if (!enabled) {
      cancelAnimationFrame(frameId); frameId = 0; pointerX = pointerY = 0;
      demo.style.setProperty('--pg-rx', '0deg'); demo.style.setProperty('--pg-ry', '0deg');
      demo.style.setProperty('--pg-front', '0px'); demo.style.setProperty('--pg-evidence', '0px'); demo.style.setProperty('--pg-shift', '0px');
    } else { lastTime = 0; requestDraw(); }
  }
  demo.addEventListener('pointermove', event => {
    if (!enabled || event.pointerType === 'touch') return;
    const bounds = demo.getBoundingClientRect();
    pointerX = clamp((event.clientX - bounds.left) / bounds.width * 2 - 1, -1, 1);
    pointerY = clamp((event.clientY - bounds.top) / bounds.height * 2 - 1, -1, 1);
    requestDraw();
  }, { passive: true });
  demo.addEventListener('pointerleave', () => { pointerX = pointerY = 0; requestDraw(); });
  addEventListener('scroll', requestDraw, { passive: true });
  addEventListener('resize', requestDraw, { passive: true });
  document.addEventListener('visibilitychange', () => { lastTime = 0; requestDraw(); });
  reducedMotion.addEventListener('change', updateMotion); precisePointer.addEventListener('change', updateMotion);
  if ('IntersectionObserver' in window) new IntersectionObserver(entries => { visible = entries[0].isIntersecting; if (visible) { lastTime = 0; requestDraw(); } }).observe(hero);
  updateMotion();
})();
