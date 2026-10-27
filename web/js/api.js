/**
 * SignSense API Client
 * Wraps fetch calls with friendly error handling and AbortController support.
 */

async function handleResponse(res) {
  if (!res.ok) {
    let detail = `Server error (${res.status})`;
    try {
      const err = await res.json();
      if (err.detail) detail = err.detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  async getHealth(signal) {
    const res = await fetch('/api/health', { signal });
    return handleResponse(res);
  },

  async getModels(signal) {
    const res = await fetch('/api/models', { signal });
    return handleResponse(res);
  },

  async getSamples(signal) {
    const res = await fetch('/api/samples', { signal });
    return handleResponse(res);
  },

  async getTeam(signal) {
    const res = await fetch('/api/team', { signal });
    return handleResponse(res);
  },

  async predict(file, model, signal) {
    const form = new FormData();
    form.append('image', file);
    form.append('model', model);
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: form,
      signal,
    });
    return handleResponse(res);
  },

  async compare(file, signal) {
    const form = new FormData();
    form.append('image', file);
    const res = await fetch('/api/compare', {
      method: 'POST',
      body: form,
      signal,
    });
    return handleResponse(res);
  },

  async attack(file, model, eps, signal) {
    const form = new FormData();
    form.append('image', file);
    form.append('model', model);
    form.append('eps', eps.toString());
    const res = await fetch('/api/attack', {
      method: 'POST',
      body: form,
      signal,
    });
    return handleResponse(res);
  },

  async sweep(file, signal) {
    const form = new FormData();
    form.append('image', file);
    const res = await fetch('/api/attack/sweep', {
      method: 'POST',
      body: form,
      signal,
    });
    return handleResponse(res);
  },

  async getReport(signal) {
    const res = await fetch('/api/report', { signal });
    return handleResponse(res);
  },
};
