import email
from email import policy
from email.message import EmailMessage

def parse_eml_content(file_bytes: bytes) -> dict:
    """
    Parses a raw .eml file byte array into a structured dictionary containing
    the sender, subject, complete raw headers, and plain-text body.
    """
    msg = email.message_from_bytes(file_bytes, policy=policy.default)
    
    sender = msg.get('From', '')
    subject = msg.get('Subject', '')
    
    # Extract all headers as a single block of text for the security analyzer
    headers_list = []
    for k, v in msg.items():
        headers_list.append(f"{k}: {v}")
    raw_headers = "\n".join(headers_list)
    
    # Extract the best plain-text body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # We only care about text/plain for the AI analyzer to read
            if content_type == "text/plain" and "attachment" not in content_disposition:
                body = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                break
    else:
        body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')

    return {
        "sender": sender,
        "subject": subject,
        "headers": raw_headers,
        "body": body.strip()
    }
