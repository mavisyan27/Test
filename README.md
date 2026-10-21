# Diabetes v1.2 Prediction API

This service exposes the exact v1.2 HistGradientBoosting model trained from `output_data.xlsx` using these six features:

- `age`
- `hypertension`
- `heart_disease`
- `bmi`
- `hba1c_level`
- `glucose`

It is an educational research prototype. It predicts the dataset's existing `diabetes` label; it does **not** predict future one-hour insulin-management risk and must not be used for diagnosis, medication, or insulin decisions.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MODEL_API_KEY='use-a-long-random-secret'
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer use-a-long-random-secret' \\
  -d '{"age":54,"hypertension":0,"heart_disease":0,"bmi":27.32,"hba1c_level":6.6,"glucose":140}'
```

## Docker

```bash
docker build -t diabetes-v12-api .
docker run --rm -p 8000:8000 -e MODEL_API_KEY='use-a-long-random-secret' diabetes-v12-api
```

Deploy this container to an HTTPS service. Base44 should call it through a server-side backend function, not directly from browser code.

## Base44 backend call

Set these Base44 server-side environment variables:

- `DIABETES_MODEL_API_URL` = the deployed API base URL, for example `https://your-domain.example`
- `DIABETES_MODEL_API_KEY` = the same secret as `MODEL_API_KEY`

Example Base44 backend function:

```ts
export default async function handler(req: Request): Promise<Response> {
  const input = await req.json();
  const response = await fetch(`${Deno.env.get("DIABETES_MODEL_API_URL")}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${Deno.env.get("DIABETES_MODEL_API_KEY")}`,
    },
    body: JSON.stringify(input),
  });
  const body = await response.text();
  return new Response(body, {
    status: response.status,
    headers: { "Content-Type": "application/json" },
  });
}
```

The Base44 frontend can invoke that function with `base44.functions.invoke("predictDiabetes", input)`.
