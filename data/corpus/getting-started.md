# Getting Started

## Create an account

Sign up at dashboard.fleetops.io. Every account starts with a 30-day trial on the Growth plan; no credit card required.

## API keys

In the dashboard, go to Settings > API keys. You will see two keys:

- `pk_live_...` — publishable key, safe for client-side code.
- `sk_live_...` — secret key, server-side only. Never expose it in browsers or mobile apps.

Use test keys (`pk_test_...`, `sk_test_...`) while integrating. Test mode never creates real shipments or charges.

## Base URL and authentication

All API requests go to `https://api.fleetops.io/v2` and must include your secret key in the `Authorization` header:

```
Authorization: Bearer sk_live_abc123
```

Requests without a valid key return `401 Unauthorized`.

## Create your first shipment

```bash
curl -X POST https://api.fleetops.io/v2/shipments \
  -H "Authorization: Bearer sk_live_abc123" \
  -H "Content-Type: application/json" \
  -d '{"to_postal": "78701", "weight_oz": 16}'
```

A successful request returns `201 Created` with a shipment object containing an id, tracking number, and label URL.
