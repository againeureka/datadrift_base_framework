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

type OperatorSpec = {
  name: string;
  version: string;
  input_count: number;
  input_types: string[];
  description: string;
  params_schema: any;
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${__API_BASE__}${path}`, init);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function isCSV(name: string) {
  return name.toLowerCase().endsWith(".csv");
}

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [operators, setOperators] = useState<OperatorSpec[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("dataset");
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const refresh = async () => {
    setErr(null);
    const [ds, ops] = await Promise.all([
      api<Dataset[]>("/datasets"),
      api<OperatorSpec[]>("/operators"),
    ]);
    setDatasets(ds);
    setOperators(ops);
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
      } catch {
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
    setSelectedIds([ds.id]);
  };

  const toggleSelect = (id: number) => {
    setSelectedIds(prev => {
      const has = prev.includes(id);
      if (has) return prev.filter(x => x !== id);
      // MVP: allow up to 2 selections (for diff). Replace oldest if >2.
      const next = [...prev, id];
      if (next.length <= 2) return next;
      return next.slice(next.length - 2);
    });
  };

  const selected = useMemo(() => datasets.filter(d => selectedIds.includes(d.id)), [datasets, selectedIds]);

  const run = async (operator_name: string, dataset_ids: number[], params: any = {}) => {
    setErr(null);
    const j = await api<Job>(`/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operator_name, dataset_ids, params }),
    });
    setJobs(prev => ({ ...prev, [j.task_id]: j }));
  };

  const runEda = async () => {
    if (selectedIds.length !== 1) return;
    await run("eda", [selectedIds[0]], {});
  };

  const runDiff = async () => {
    if (selectedIds.length !== 2) return;
    const a = selected[0];
    const b = selected[1];
    const bothCsv = a && b && isCSV(a.original_filename) && isCSV(b.original_filename);
    await run(bothCsv ? "diff.tabular" : "diff.file", selectedIds, {});
  };

  const operatorHelp = useMemo(() => {
    const byName: Record<string, OperatorSpec> = {};
    for (const o of operators) byName[o.name] = o;
    return byName;
  }, [operators]);

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: 20, maxWidth: 1100, margin: "0 auto" }}>
      <h1 style={{ margin: "8px 0 16px" }}>ddoc MVP v0.2</h1>
      <p style={{ marginTop: 0, opacity: 0.8 }}>
        Upload → cards → run typed operators (EDA / DIFF) via worker.
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
            Select up to 2 cards. One card: EDA. Two cards: DIFF (CSV→tabular diff else file diff).
          </p>
        </div>

        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 14 }}>
          <h2 style={{ marginTop: 0 }}>Actions</h2>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button onClick={runEda} disabled={selectedIds.length !== 1}>Run EDA</button>
            <button onClick={runDiff} disabled={selectedIds.length !== 2}>Run DIFF</button>
          </div>

          <div style={{ marginTop: 10, opacity: 0.85 }}>
            <div><b>Selected:</b> {selectedIds.length === 0 ? "none" : selectedIds.join(", ")}</div>
            {selected.length > 0 && (
              <ul style={{ marginTop: 6 }}>
                {selected.map(s => (
                  <li key={s.id}>
                    #{s.id} {s.name} — {s.original_filename}
                    {s.download_url && (
                      <> — <a href={s.download_url} target="_blank" rel="noreferrer">download</a></>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <details style={{ marginTop: 8 }}>
            <summary>Available operators</summary>
            <pre style={{ whiteSpace: "pre-wrap", background: "#f7f7f7", padding: 10, borderRadius: 10, marginTop: 8, maxHeight: 240, overflow: "auto" }}>
{JSON.stringify(operators, null, 2)}
            </pre>
          </details>
        </div>
      </div>

      <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 14 }}>
          <h2 style={{ marginTop: 0 }}>Dataset cards</h2>
          <div style={{ display: "grid", gap: 10 }}>
            {datasets.map(ds => {
              const selected = selectedIds.includes(ds.id);
              return (
                <div
                  key={ds.id}
                  onClick={() => toggleSelect(ds.id)}
                  style={{
                    cursor: "pointer",
                    border: selected ? "2px solid #333" : "1px solid #ddd",
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
              );
            })}
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
                {j.result?.result?.report?.summary && (
                  <pre style={{ whiteSpace: "pre-wrap", background: "#f7f7f7", padding: 10, borderRadius: 10, marginTop: 8, maxHeight: 260, overflow: "auto" }}>
{JSON.stringify(j.result.result.report.summary, null, 2)}
                  </pre>
                )}
              </div>
            ))}
            {Object.keys(jobs).length === 0 && <div style={{ opacity: 0.7 }}>No jobs yet. Run EDA/DIFF.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
