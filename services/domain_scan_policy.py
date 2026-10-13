"""Scope domain scans to submitted hosts; never infer whole-domain ownership."""
from copy import deepcopy
import ipaddress
import re
from urllib.parse import urlsplit

def domain_scan_input(value):
    raw=value.strip()
    if not raw or any(c.isspace() or ord(c)<32 for c in raw):
        raise ValueError('Enter a hostname, without spaces.')
    parsed=urlsplit(raw if '://' in raw else 'https://'+raw)
    if parsed.scheme.lower() not in ('http','https'):
        raise ValueError('Use an HTTP(S) hostname.')
    if parsed.username or parsed.password or parsed.port or parsed.path not in ('','/') or parsed.query or parsed.fragment:
        raise ValueError('Use the URL scanner for links containing a path, query, credentials or port.')
    host=(parsed.hostname or '').encode('idna').decode('ascii').lower().rstrip('.')
    if not host or len(host)>253:
        raise ValueError('Enter a valid hostname.')
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if '.' not in host or any(not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?',p) for p in host.split('.')):
            raise ValueError('Enter a valid hostname.')
    authority='['+host+']' if ':' in host else host
    return host,parsed.scheme.lower()+'://'+authority

def scope_domain_result(result,submitted_host):
    output=deepcopy(result)
    models=output.get('origintrace_pillars',{}).get('model_probabilities',{})
    scope=dict(submitted_hostname=submitted_host,assessed_url=output.get('resolved_url') or output.get('url'),
               task='submitted host/homepage URL assessment',
               whole_domain_ownership_verified=False,
               limitation='A homepage scan does not assess every subdomain or path, or establish domain ownership.')
    output['scan_scope']=scope
    output['domain_model_role']='diagnostic_only_unverified_domain_labels'
    # A low score without the actual classifier or retrieved page is not a clean result.
    missing=[]
    if models.get('url') is None:missing.append('URL model result unavailable')
    if not output.get('html_analysis',{}).get('success'):missing.append('Homepage content unavailable')
    output['evidence_gaps']=missing
    if missing and output.get('status') not in ('PHISHING','SUSPICIOUS') and output.get('verdict_state')!='PHISHING':
        output.update(status='UNKNOWN',verdict_state='INSUFFICIENT_EVIDENCE',risk_level='unknown',
                      summary='The completed checks cannot establish a low-risk result for this host. '+ '; '.join(missing)+'.')
    elif output.get('verdict_state')=='INSUFFICIENT_EVIDENCE' and output.get('status')=='CLEAN':
        output.update(status='UNKNOWN',risk_level='unknown',
                      summary='The completed checks leave the assessed host/homepage uncertain. Further review is required.')
    elif output.get('status')=='CLEAN':
        output['summary']='No threat found in the completed checks for the submitted host/homepage. Other paths and subdomains were not assessed.'
    elif output.get('status')=='PHISHING' and 'Custom Security Rule' not in output.get('summary',''):
        output['summary']='High-risk indicators were found for the assessed host/homepage. Avoid visiting until reviewed; this does not establish whole-domain ownership.'
    # Do not overwrite a threat finding or a configured block because another check failed.
    return output
