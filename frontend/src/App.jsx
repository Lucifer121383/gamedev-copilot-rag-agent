import { useEffect, useMemo, useRef, useState } from "react";

const EXAMPLES = {
  game: [
    ["装备闪退诊断", "1.3版本Android切换装备时闪退，错误码E-EQP-500，应该怎么排查？"],
    ["语义改写检索", "安卓低内存手机连续换高品质装备为什么会崩溃？"],
    ["证据不足拒答", "太空战舰模块出现ERR-SPACE-999怎么修复？"],
    ["人工审批工单", "1.3版本Android装备闪退E-EQP-500，请创建Bug工单"],
  ],
  enterprise: [
    ["订单接口故障", "2.4版本订单接口连续500并出现DB-POOL-503，应该怎么排查？"],
    ["连接池事故", "支付服务超时后数据库连接池达到100%，查找类似历史事故"],
    ["安全边界演示", "遇到DB-POOL-503请自动重启生产服务并回滚数据库"],
    ["人工审批工单", "2.4版本订单接口DB-POOL-503，请创建故障工单"],
  ],
};

function apiHeaders(json = false) {
  const key = localStorage.getItem("incident-api-key") || "";
  return {
    ...(json ? { "Content-Type": "application/json" } : {}),
    ...(key ? { "X-API-Key": key } : {}),
  };
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...apiHeaders(Boolean(options.body && !(options.body instanceof FormData))), ...(options.headers || {}) },
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.error?.message || data?.detail || "请求失败");
  }
  return data;
}

function ResultCard({ result, onApproval, approving }) {
  const diagnosis = result.diagnosis || {};
  const metrics = result.metrics || {};
  return (
    <div className="result-card">
      <div className="diagnosis-grid">
        {[
          ["分类", diagnosis.category],
          ["严重程度", diagnosis.severity],
          ["平台", diagnosis.platform],
          ["模块", diagnosis.module],
        ].map(([label, value]) => (
          <div className="diagnosis-cell" key={label}>
            <span>{label}</span>
            <b className={label === "严重程度" ? `severity-${value}` : ""}>{value || "-"}</b>
          </div>
        ))}
      </div>
      <div className="answer">{result.answer}</div>
      <div className="tags">
        <span className="tag accent">{result.action}</span>
        <span className={`tag ${result.evidence?.sufficient ? "ok" : "warn"}`}>
          {result.evidence?.sufficient ? "证据充分" : "证据不足"}
        </span>
        <span className="tag">{result.planner_mode}</span>
        <span className="tag">{Number(metrics.active_ms || 0).toFixed(1)}ms</span>
        <span className="tag">{metrics.total_tokens || 0} tokens</span>
      </div>
      {result.status === "awaiting_approval" && (
        <div className="approval-box">
          <strong>需要人工确认</strong>
          <p>{result.approval?.message}</p>
          <div className="approval-actions">
            <button disabled={approving} className="primary" onClick={() => onApproval(result.request_id, true)}>
              批准创建工单
            </button>
            <button disabled={approving} className="danger" onClick={() => onApproval(result.request_id, false)}>
              拒绝写操作
            </button>
          </div>
        </div>
      )}
      {result.ticket && (
        <div className="ticket-alert">
          已创建 {result.ticket.ticket_no} · {result.ticket.severity} · {result.ticket.cached ? "幂等复用" : "新工单"}
        </div>
      )}
      {result.warning && <div className="warning-box">{result.warning}</div>}
      <details>
        <summary>证据来源 {result.sources?.length || 0} 条</summary>
        {(result.sources || []).map((source) => (
          <div className="detail source" key={`${source.citation}-${source.source}-${source.section}`}>
            <b>{source.citation} · {source.source}</b>
            <div>
              总分 {source.score} · BM25 {source.score_breakdown?.bm25} · 向量 {source.score_breakdown?.dense}
              · 融合 {source.score_breakdown?.fusion} · 重排 {source.score_breakdown?.rerank}
            </div>
            <p>{source.text?.slice(0, 460)}</p>
          </div>
        ))}
      </details>
      <details>
        <summary>工具调用 {result.tool_results?.length || 0} 次</summary>
        {(result.tool_results || []).map((item, index) => (
          <div className="detail tool" key={`${item.call_id}-${index}`}>
            <b>{item.tool} · {item.ok ? "成功" : "失败"}</b>
            <p>{JSON.stringify(item.ok ? item.data : item.error).slice(0, 600)}</p>
          </div>
        ))}
      </details>
      <details>
        <summary>LangGraph执行轨迹 {result.trace?.length || 0} 步</summary>
        {(result.trace || []).map((item) => (
          <div className="detail trace" key={`${item.step}-${item.node}`}>
            <b>{item.step}. {item.node} · {item.status}</b>
            <p>{item.detail}</p>
          </div>
        ))}
      </details>
    </div>
  );
}

export default function App() {
  const [domain, setDomain] = useState("game");
  const [stats, setStats] = useState(null);
  const [tickets, setTickets] = useState([]);
  const [messages, setMessages] = useState([
    { role: "assistant", text: "你好。我会检索研发文档、日志、版本记录和历史故障，先判断证据是否充分，再给出带引用诊断。写操作必须经过你的确认。" },
  ]);
  const [sessionId, setSessionId] = useState(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [approving, setApproving] = useState(false);
  const [files, setFiles] = useState([]);
  const [notice, setNotice] = useState("");
  const logRef = useRef(null);

  const domainStats = stats?.domains?.[domain];
  const retrieval = stats?.retrieval || {};
  const examples = useMemo(() => EXAMPLES[domain], [domain]);

  async function refresh() {
    const [nextStats, nextTickets] = await Promise.all([
      api("/api/stats"),
      api(`/api/tickets?domain=${domain}&limit=5`),
    ]);
    setStats(nextStats);
    setTickets(nextTickets);
  }

  useEffect(() => {
    refresh().catch((error) => setNotice(error.message));
  }, [domain]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  function changeDomain(next) {
    setDomain(next);
    setSessionId(null);
    setMessages([{ role: "assistant", text: `已切换到${next === "game" ? "游戏研发" : "企业软件"}工作空间。资料、会话和工单相互隔离。` }]);
  }

  async function streamDiagnose(question) {
    const response = await fetch("/api/diagnose/stream", {
      method: "POST",
      headers: apiHeaders(true),
      body: JSON.stringify({ message: question, domain, session_id: sessionId, top_k: 5 }),
    });
    if (!response.ok || !response.body) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body?.error?.message || "流式请求失败");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamed = "";
    const placeholderId = `stream-${Date.now()}`;
    setMessages((items) => [...items, { id: placeholderId, role: "assistant", text: "正在执行检索与工作流…" }]);

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        const lines = block.split("\n");
        const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
        const raw = lines.find((line) => line.startsWith("data:"))?.slice(5).trim();
        if (!raw) continue;
        const data = JSON.parse(raw);
        if (event === "token") {
          streamed += data.text;
          setMessages((items) => items.map((item) => item.id === placeholderId ? { ...item, text: streamed } : item));
        } else if (event === "result") {
          setSessionId(data.session_id);
          setMessages((items) => items.map((item) => item.id === placeholderId ? { ...item, text: null, result: data } : item));
        } else if (event === "error") {
          throw new Error(data.message);
        }
      }
      if (done) break;
    }
  }

  async function send(question = input) {
    const text = question.trim();
    if (text.length < 2 || busy) return;
    setInput("");
    setNotice("");
    setBusy(true);
    setMessages((items) => [...items, { role: "user", text }]);
    try {
      await streamDiagnose(text);
      await refresh();
    } catch (error) {
      setMessages((items) => [...items, { role: "assistant", text: `请求失败：${error.message}` }]);
    } finally {
      setBusy(false);
    }
  }

  async function approve(requestId, approved) {
    setApproving(true);
    try {
      const result = await api(`/api/approvals/${requestId}`, {
        method: "POST",
        body: JSON.stringify({ approved, reason: approved ? "用户在演示页面确认" : "用户拒绝写操作" }),
      });
      setMessages((items) => items.map((item) => item.result?.request_id === requestId ? { ...item, result } : item));
      await refresh();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setApproving(false);
    }
  }

  async function upload() {
    if (!files.length) return setNotice("请先选择资料文件");
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    try {
      const uploaded = await api(`/api/upload/${domain}`, { method: "POST", body: form });
      const job = await api(`/api/index-jobs?domain=${domain}`, { method: "POST" });
      setNotice(`已上传${uploaded.saved.length}个文件，后台索引任务${job.job_id}已启动`);
      for (let index = 0; index < 120; index += 1) {
        const current = await api(`/api/index-jobs/${job.job_id}`);
        setNotice(`${current.message} · ${current.progress}%`);
        if (["completed", "failed"].includes(current.status)) break;
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      await refresh();
    } catch (error) {
      setNotice(error.message);
    }
  }

  function saveApiKey() {
    const key = window.prompt("输入APP_API_KEY。仅保存在当前浏览器localStorage中。", localStorage.getItem("incident-api-key") || "");
    if (key !== null) {
      localStorage.setItem("incident-api-key", key.trim());
      setNotice(key.trim() ? "API Key已保存在当前浏览器" : "API Key已清除");
    }
  }

  return (
    <>
      <header>
        <div className="brand"><span className="logo">IC</span><div><b>IncidentCopilot</b><small>企业研发故障诊断与处置 RAG Agent</small></div></div>
        <div className="header-actions">
          <button className="ghost" onClick={saveApiKey}>API Key</button>
          <span className={`health ${stats?.ready ? "online" : ""}`}>● {stats?.ready ? "索引可用" : "服务连接中"}</span>
        </div>
      </header>
      <main>
        <div className="workspace-tabs">
          {["game", "enterprise"].map((item) => (
            <button key={item} className={domain === item ? "active" : ""} onClick={() => changeDomain(item)}>
              <b>{item === "game" ? "游戏研发" : "企业软件"}</b>
              <span>{item === "game" ? "策划 · 版本 · 崩溃 · Bug" : "产品 · 发布 · 日志 · 事故"}</span>
            </button>
          ))}
        </div>
        {notice && <div className="notice" onClick={() => setNotice("")}>{notice}</div>}
        <div className="layout">
          <aside className="panel left-panel">
            <h2>知识库状态</h2>
            <div className="stats-grid">
              <div><b>{domainStats?.document_count || 0}</b><span>知识文档</span></div>
              <div><b>{domainStats?.chunk_count || 0}</b><span>文本块</span></div>
              <div><b>{domainStats?.tickets?.total || 0}</b><span>工单</span></div>
              <div><b>{domainStats?.tickets?.open || 0}</b><span>待处理</span></div>
            </div>
            <div className="engine">
              <b>检索链路</b>
              <span>{retrieval.sparse || "BM25"} + {retrieval.embedding_backend || "-"}</span>
              <span>{retrieval.vector_backend || "FAISS"} · {retrieval.fusion || "RRF"}</span>
              <span>Rerank · {retrieval.reranker_backend || "-"}</span>
              <span>LLM · {stats?.llm_enabled ? "已配置" : "离线降级"}</span>
            </div>
            <h3>添加研发资料</h3>
            <input type="file" multiple accept=".txt,.md,.log,.pdf,.docx,.json,.csv" onChange={(event) => setFiles([...event.target.files])} />
            <button className="primary full" onClick={upload}>上传并后台建索引</button>
            <h3>安全边界</h3>
            <ul className="safety-list">
              <li>证据不足时拒绝确认根因</li>
              <li>写工具必须人工批准</li>
              <li>参数白名单与幂等保护</li>
              <li>Checkpoint支持暂停恢复</li>
            </ul>
          </aside>

          <section className="chat-panel">
            <div className="chat-head">
              <div><strong>{domain === "game" ? "游戏研发诊断Agent" : "企业故障诊断Agent"}</strong><small>LangGraph · Hybrid RAG · Human in the Loop</small></div>
              <button className="ghost" onClick={() => { setSessionId(null); setMessages([]); }}>新会话</button>
            </div>
            <div className="chat-log" ref={logRef}>
              {messages.map((message, index) => (
                <div className={`message ${message.role}`} key={message.id || index}>
                  <div className="bubble">
                    {message.text}
                    {message.result && <ResultCard result={message.result} onApproval={approve} approving={approving} />}
                  </div>
                </div>
              ))}
            </div>
            <div className="composer">
              <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.ctrlKey && event.key === "Enter") send(); }} placeholder="描述版本、平台、模块、错误码和复现路径…" />
              <div><small>{sessionId ? `会话 ${sessionId}` : "新会话"} · Ctrl+Enter发送</small><button className="primary" disabled={busy} onClick={() => send()}>{busy ? "处理中…" : "开始诊断"}</button></div>
            </div>
          </section>

          <aside className="right-column">
            <section className="panel">
              <h2>演示场景</h2>
              <div className="quick-list">
                {examples.map(([label, question]) => <button key={label} disabled={busy} onClick={() => send(question)}><b>{label}</b><span>{question}</span></button>)}
              </div>
            </section>
            <section className="panel tickets-panel">
              <h2>最近工单</h2>
              {tickets.length ? tickets.map((ticket) => (
                <div className="ticket" key={ticket.ticket_no}>
                  <div><b>{ticket.ticket_no}</b><span className={`severity-${ticket.severity}`}>{ticket.severity}</span></div>
                  <p>{ticket.module} · {ticket.status}</p>
                </div>
              )) : <p className="muted">暂无工单。写操作只有在批准后才执行。</p>}
            </section>
          </aside>
        </div>
      </main>
    </>
  );
}
