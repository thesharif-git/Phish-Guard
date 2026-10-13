(() => {
  const form = document.querySelector('form[data-mode]');
  if (!form) return;
  const signup = form.dataset.mode === 'register';
  const password = form.querySelector('#password');
  const toggle = form.querySelector('.password-toggle');
  toggle.addEventListener('click', () => {
    const visible = password.type === 'password';
    password.type = visible ? 'text' : 'password';
    toggle.setAttribute('aria-pressed', String(visible));
    toggle.setAttribute('aria-label', visible ? 'Hide password' : 'Show password');
  });
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = form.querySelector('button[type="submit"]');
    if (button.disabled) return;
    const error = document.getElementById('authError');
    const success = document.getElementById('authSuccess');
    const label = signup ? 'Create account' : 'Sign in';
    button.disabled = true;
    button.textContent = signup ? 'Creating account…' : 'Signing in…';
    form.setAttribute('aria-busy', 'true');
    error.hidden = true;
    success.hidden = true;
    try {
      const data = new FormData(form);
      const options = signup
        ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(Object.fromEntries(data.entries()))}
        : {method:'POST', body:data};
      const response = await fetch(signup ? '/auth/register' : '/auth/login', options);
      const result = await response.json();
      if (!response.ok) {
        const detail = Array.isArray(result.detail) ? result.detail[0]?.msg : result.detail;
        throw new Error(typeof detail === 'string' ? detail : 'Please check your details and try again.');
      }
      if (signup) {
        success.textContent = 'Account created. Taking you to sign in…';
        success.hidden = false;
        window.setTimeout(() => window.location.assign('/login'), 1500);
      } else {
        const next = new URL(document.getElementById('nextUrl')?.value || '/dashboard', window.location.origin);
        window.location.assign(next.origin === window.location.origin ? next.href : '/dashboard');
      }
    } catch (failure) {
      error.textContent = failure instanceof TypeError ? 'Unable to connect. Please try again.' : failure.message;
      error.hidden = false;
      button.disabled = false;
      button.textContent = label;
    } finally {
      form.removeAttribute('aria-busy');
    }
  });
})();