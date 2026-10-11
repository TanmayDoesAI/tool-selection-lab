"""One POST helper with standard retry: 408/409/429/5xx and timeouts back off and retry; other errors fail fast."""
import random,time
import httpx

RETRYABLE={408,409,425,429,500,502,503,504,529}

class APIError(RuntimeError):pass

def retry_after(response):
    value=response.headers.get('retry-after')
    try:return float(value) if value is not None else None
    except ValueError:return None

def post_json(client,url,headers,body,retries=6,max_backoff=60.0,sleep=time.sleep):
    """POST JSON and return the decoded response. Waits honour Retry-After (capped), else 1, 2, 4 ... s plus jitter."""
    for attempt in range(retries+1):
        delay=None
        try:response=client.post(url,headers=headers,json=body)
        except (httpx.TimeoutException,httpx.TransportError) as exc:error=f'{type(exc).__name__} calling {url}'
        else:
            if response.is_success:return response.json()
            error=f'HTTP {response.status_code} from {url}: {response.text[:300]}'
            if response.status_code not in RETRYABLE:raise APIError(error)
            delay=retry_after(response)
        if attempt==retries:raise APIError(f'{error} (gave up after {retries+1} attempts)')
        sleep(min(max_backoff,delay if delay is not None else 2**attempt+random.uniform(0,1)))
