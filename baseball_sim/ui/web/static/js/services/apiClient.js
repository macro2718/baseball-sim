// Lightweight wrapper around fetch that normalises JSON responses.

export async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    const err = new Error('Request failed');
    err.cause = error;
    err.payload = null;
    throw err;
  }

  if (!response.ok) {
    const err = new Error(payload?.error || 'Request failed');
    err.payload = payload || null;
    throw err;
  }

  return payload;
}
