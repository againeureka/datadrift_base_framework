import React, { useEffect, useMemo, useState } from "react";

declare const __API_BASE__: string;

type Dataset = {
  id: number;
  name: string;
  original_filename: string;
  created_at: string;
  download_url?: string | null;
};

type Job = {
  task_id: string;
  state: string;
  result?: any;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${__API_BASE__}${path}`, init);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("dataset");
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [selected, setSelected] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = async () => {
    setErr(null);
    const ds = await api<Dataset[]>("/datasets");
    setDatasets(ds);
  };

  useEffect(() => {
    refresh().catch(e => setErr(String(e)));
  }, []);

  // poll jobs
  useEffect(() => {
    const ids = Object.keys(jobs);
    if (ids.length === 0) return;
    const t = setInterval(async () => {
      try {
        const updates: Record<string, Job> = { ...jobs };
        for (const id of ids) {
          const j = await api<Job>(`/jobs/${id}`);
          updates[id] = j;
        }
        setJobs(updates);
      } catch (e) {
        // ignore transient errors
      }
    }, 1500);
    return () => clearInterval(t);
  }, [JSON.stringify(Object.keys(jobs))]); // eslint-disable-line react-hooks/exhaustive-deps

  const onUpload = async () => {
    if (!file) return;
    setErr(null);
    const form = new FormData();
    form.append("file", file);
    const ds = await api<Dataset>(`/datasets?name=${encodeURIComponent(name)}`, {
      method: "POST",
      body: form,
    });
    setFile(null);
    await refresh();
    setSelected(ds.id);
  };

  const runEda = async (datasetId: number) => {
    setErr(null);
    const j = await api<Job>(`/datasets/${datasetId}/eda`, { method: "POST" });
    setJobs(prev => ({ ...prev, [j.task_id]: j }));
  };

  const selectedDs = datasets.find(d => d.id === selected) || null;

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 20, maxWidth: 1100, margin: "0 auto" }}>
      <h1 style={{ margin: "8px 0 16px" }}>ddoc MVP</h1>
      <p style={{ marginTop: 0, opacity: 0.8 }}>
        Upload → dataset card → run EDA (worker) → view report.
      </p>

      {err && (
        <div style={{ padding: 12, border: "1px solid #f99", background: "#fff5f5", marginBottom: 12 }}>
          {err}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 14 }}>
          <h2 style={{ marginTop: 0 }}>Upload</h2>
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="dataset name" />
            <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            <button onClick={onUpload} disabled={!file}>Upload</button>
            <button onClick={() => refresh()} style={{ marginLeft: "auto" }}>Refresh</button>
          </div>
          <p style={{ opacity: 0.75, marginBottom: 0 }}>
            CSV gets basic EDA in MVP. ZIP/others get file metadata. Extend via adapters/operators.
          </p>
        </div>

        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 14 }}>
          <h2 style={{ marginTop: 0 }}>Selected dataset</h2>
          {selectedDs ? (
            <>
              <div><b>{selectedDs.name}</b> <span style={{ opacity: 0.7 }}>#{selectedDs.id}</span></div>
              <div style={{ opacity: 0.85 }}>{selectedDs.original_filename}</div>
              <div style={{ opacity: 0.7, fontSize: 13 }}>{new Date(selectedDs.created_at).toLocaleString()}</div>
              <div style={{ marginTop: 10, display: "flex", gap: 10 }}>
                <button onClick={() => runEda(selectedDs.id)}>Run EDA</button>
                {selectedDs.download_url && (
                  <a href={selectedDs.download_url} target="_blank" rel="noreferrer">Download (presigned)</a>
                )}
              </div>
            </>
          ) : (
            <div style={{ opacity: 0.7 }}>Select a dataset card from the list.</div>
          )}
        </div>
      </div>

      <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 14 }}>
          <h2 style={{ marginTop: 0 }}>Dataset cards</h2>
          <div style={{ display: "grid", gap: 10 }}>
            {datasets.map(ds => (
              <div
                key={ds.id}
                onClick={() => setSelected(ds.id)}
                style={{
                  cursor: "pointer",
                  border: ds.id === selected ? "2px solid #333" : "1px solid #ddd",
                  borderRadius: 12,
                  padding: 12
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                  <b>{ds.name}</b>
                  <span style={{ opacity: 0.6, fontSize: 13 }}>#{ds.id}</span>
                </div>
                <div style={{ opacity: 0.8 }}>{ds.original_filename}</div>
                <div style={{ opacity: 0.6, fontSize: 12 }}>{new Date(ds.created_at).toLocaleString()}</div>
              </div>
            ))}
            {datasets.length === 0 && <div style={{ opacity: 0.7 }}>No datasets yet. Upload one.</div>}
          </div>
        </div>

        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 14 }}>
          <h2 style={{ marginTop: 0 }}>Jobs</h2>
          <div style={{ display: "grid", gap: 10 }}>
            {Object.values(jobs).map(j => (
              <div key={j.task_id} style={{ border: "1px solid #ddd", borderRadius: 12, padding: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <b>{j.task_id.slice(0, 8)}…</b>
                  <span style={{ opacity: 0.75 }}>{j.state}</span>
                </div>
                {j.result?.report_url && (
                  <div style={{ marginTop: 8 }}>
                    <a href={j.result.report_url} target="_blank" rel="noreferrer">Open report (JSON)</a>
                  </div>
                )}
                {j.result?.report?.summary && (
                  <pre style={{ whiteSpace: "pre-wrap", background: "#f7f7f7", padding: 10, borderRadius: 10, marginTop: 8, maxHeight: 260, overflow: "auto" }}>
{JSON.stringify(j.result.report.summary, null, 2)}
                  </pre>
                )}
              </div>
            ))}
            {Object.keys(jobs).length === 0 && <div style={{ opacity: 0.7 }}>No jobs yet. Run EDA.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
