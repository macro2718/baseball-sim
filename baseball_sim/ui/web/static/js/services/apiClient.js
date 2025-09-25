// Lightweight wrapper around fetch that normalises JSON responses.

export async function apiRequest(url, options = {}) {
  const hasBody = options && Object.prototype.hasOwnProperty.call(options, 'body') && options.body != null;
  const method = (options && options.method ? String(options.method) : 'GET').toUpperCase();
  const headers = new Headers(options && options.headers ? options.headers : undefined);
  // Only set JSON content-type when sending a body
  if (hasBody && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(url, {
    ...options,
    method,
    headers,
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
