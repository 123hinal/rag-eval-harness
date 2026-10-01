# Webhooks

Webhooks let FleetOps notify your systems in real time when shipment events occur.

## Event types

- `shipment.created`
- `shipment.in_transit`
- `shipment.out_for_delivery`
- `shipment.delivered`
- `shipment.exception`
- `return.delivered`

## Configuring an endpoint

Add your HTTPS endpoint under Settings > Webhooks in the dashboard and select the events you want to receive. Endpoints must respond within 10 seconds.

## Verifying signatures

Every webhook request includes an `X-FleetOps-Signature` header: an HMAC-SHA256 hex digest of the raw request body, computed with your webhook secret. Always verify the signature before acting on an event:

```python
import hmac, hashlib

expected = hmac.new(webhook_secret.encode(), raw_body, hashlib.sha256).hexdigest()
if not hmac.compare_digest(expected, signature_header):
    raise ValueError("invalid webhook signature")
```

## Retries

If your endpoint returns a non-2xx status or times out, FleetOps retries delivery 5 times with exponential backoff over 24 hours. After the final failure the event is marked dead and visible in the dashboard's webhook log.
