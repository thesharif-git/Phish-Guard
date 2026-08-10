// Global JS utilities used on all pages

// ── Theme System ───────────────────────────────────────────────
function updateThemeIcon(theme) {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    const icon = btn.querySelector('.material-symbols-outlined');
    if (icon) {
        icon.textContent = theme === 'dark' ? 'light_mode' : 'dark_mode';
    }
    btn.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
}

function initTheme() {
    const saved = localStorage.getItem('pg-theme') || 'light';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeIcon(saved);
}

function toggleTheme(override) {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = override || (current === 'dark' ? 'light' : 'dark');
    
    // Add temporary class to enforce fully synchronized transitions across all elements
    document.body.classList.add('theme-transitioning');
    
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('pg-theme', next);
    updateThemeIcon(next);
    
    setTimeout(() => {
        document.body.classList.remove('theme-transitioning');
    }, 250); // Remove after transition finishes (200ms + buffer)

    // Save to profile if logged in
    fetch('/api/profile', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ theme: next })
    }).catch(() => {});
}

// Initialize theme on every dashboard page load
initTheme();

// ── Tab switching utility ──────────────────────────────────────
function switchTab(id, btn) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + id).classList.add('active');
}

// ── Table filter utility ───────────────────────────────────────
function filterTable(input, tableId) {
    const q = input.value.toLowerCase();
    document.querySelectorAll(`#${tableId} tbody tr`).forEach(row => {
        row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
}

// ── Global Authentication ──────────────────────────────────────
async function logoutUser() {
    try {
        await fetch('/auth/logout', { method: 'POST' });
        window.location.href = '/login';
    } catch (err) {
        console.error("Logout failed:", err);
        window.location.href = '/login';
    }
}

// ── Live Clock ─────────────────────────────────────────────────
(function clockTick() {
    const el = document.getElementById('liveClock');
    if (el) {
        const now = new Date();
        el.textContent = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
            + ' · ' + now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    }
    setTimeout(clockTick, 30000);
})();

// ── Notifications System ───────────────────────────────────────
let _notifOpen = false;

async function loadNotifications() {
    try {
        const res = await fetch('/api/notifications');
        if (!res.ok) return;
        const data = await res.json();
        const list = document.getElementById('notifList');
        const badge = document.getElementById('notifBadge');
        if (!list) return;

        const notifs = data.notifications || [];
        const unread = notifs.filter(n => !n.is_read).length;

        if (badge) {
            badge.textContent = unread;
            badge.classList.toggle('hidden', unread === 0);
        }

        if (notifs.length === 0) {
            list.innerHTML = '<div class="px-4 py-6 text-center text-xs text-muted">No notifications yet.<br/>Phishing alerts will appear here.</div>';
            return;
        }

        const levelClasses = {
            critical: 'bg-red-500/10 border-red-500/20 text-red-400',
            warning:  'bg-yellow-500/10 border-yellow-500/20 text-yellow-400',
            info:     'bg-blue-500/10 border-blue-500/20 text-blue-400',
        };

        list.innerHTML = notifs.map(n => `
            <a href="${n.scan_id ? '/email/' + n.scan_id : '#'}" class="block px-4 py-3 hover:bg-panel2/50 transition-colors ${n.is_read ? 'opacity-60' : ''}">
                <div class="flex items-start gap-2">
                    <span class="text-xs px-1.5 py-0.5 rounded border ${levelClasses[n.level] || levelClasses.info} whitespace-nowrap font-medium">${n.level}</span>
                    <div class="flex-1 min-w-0">
                        <p class="text-xs text-white font-medium truncate">${n.title}</p>
                        <p class="text-[10px] text-muted mt-0.5 leading-relaxed">${n.message}</p>
                        <p class="text-[10px] text-muted/60 mt-1">${n.created_at}</p>
                    </div>
                    ${!n.is_read ? '<div class="w-1.5 h-1.5 rounded-full bg-info flex-shrink-0 mt-1"></div>' : ''}
                </div>
            </a>
        `).join('');
    } catch(e) {
        const list = document.getElementById('notifList');
        if (list) list.innerHTML = '<div class="px-4 py-6 text-center text-xs text-muted">Unable to load notifications.<br/>Check your connection and try again.</div>';
    }
}

function toggleNotifications() {
    const dd = document.getElementById('notifDropdown');
    if (!dd) return;
    _notifOpen = !_notifOpen;
    dd.classList.toggle('hidden', !_notifOpen);
    if (_notifOpen) loadNotifications();
}

async function markAllRead() {
    await fetch('/api/notifications/read-all', { method: 'PATCH' });
    loadNotifications();
}

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
    const wrapper = document.getElementById('notifWrapper');
    if (wrapper && !wrapper.contains(e.target) && _notifOpen) {
        _notifOpen = false;
        document.getElementById('notifDropdown').classList.add('hidden');
    }
});

// Load badge count on page load
loadNotifications();

// ── Global Header Search ────────────────────────────────────────
(function wireHeaderSearch() {
    const input = document.querySelector('header input[type="text"]');
    if (!input) return;
    input.addEventListener('input', () => {
        const q = input.value.toLowerCase();
        const table = document.getElementById('scansTable') || document.querySelector('table');
        if (!table) return;
        table.querySelectorAll('tbody tr').forEach(row => {
            row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
        });
        const hint = document.getElementById('searchEmptyHint');
        if (hint) {
            const visible = table.querySelectorAll('tbody tr:not([style*="none"])').length;
            hint.classList.toggle('hidden', visible > 0 || !q);
        }
    });
})();

// ── Mobile Sidebar Toggle ─────────────────────────────────────
let _mobileSidebarOpen = false;
function toggleMobileSidebar() {
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebarBackdrop');
    if (!sidebar || !backdrop) return;

    _mobileSidebarOpen = !_mobileSidebarOpen;
    if (_mobileSidebarOpen) {
        sidebar.classList.remove('-translate-x-full');
        sidebar.classList.add('translate-x-0');
        backdrop.classList.remove('opacity-0', 'pointer-events-none');
        backdrop.classList.add('opacity-100', 'pointer-events-auto');
        document.body.style.overflow = 'hidden';
    } else {
        sidebar.classList.add('-translate-x-full');
        sidebar.classList.remove('translate-x-0');
        backdrop.classList.add('opacity-0', 'pointer-events-none');
        backdrop.classList.remove('opacity-100', 'pointer-events-auto');
        document.body.style.overflow = '';
    }
}

// ── Copy to Clipboard Helper ──────────────────────────────────
async function copyToClipboard(text, btnElement) {
    try {
        await navigator.clipboard.writeText(text);
        if (btnElement) {
            const originalIcon = btnElement.innerHTML;
            btnElement.innerHTML = '<span class="material-symbols-outlined text-[14px]">check</span>';
            btnElement.classList.add('text-emerald-500');
            setTimeout(() => {
                btnElement.innerHTML = originalIcon;
                btnElement.classList.remove('text-emerald-500');
            }, 2000);
        }
        showToast('Copied', 'Copied to clipboard', 'success');
    } catch (err) {
        showToast('Error', 'Failed to copy', 'error');
    }
}
