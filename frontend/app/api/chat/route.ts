export async function POST(req: Request) {
  const body = await req.json();

  const backendUrl = process.env.RAG_BACKEND_URL || "http://localhost:8765";

  const response = await fetch(`${backendUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok || !response.body) {
    return new Response("Backend error", { status: 502 });
  }

  return new Response(response.body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
}
