# Rate Shopping

## Get rates

```
POST /v2/rates
```

Request body:

```json
{
  "from_postal": "78701",
  "to_postal": "10001",
  "weight_oz": 16,
  "length_in": 10,
  "width_in": 8,
  "height_in": 6
}
```

The response lists carrier services sorted by price, each with `service_name`, `total_price`, and `estimated_days`.

## Negotiated rates

Connect your own carrier accounts (UPS, FedEx, DHL, USPS) under Settings > Carriers to see your negotiated discounts alongside FleetOps' pre-negotiated rates. Your accounts are used automatically whenever they beat the default pricing.

## Rate caching

Identical rate requests are cached for 15 minutes. Send an idempotency key with the request if you need to bypass the cache for a fresh quote.
