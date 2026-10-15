import { useState, useCallback, useEffect } from 'react';
import { analysis } from '../api';
import AgentTrace from '../components/AgentTrace.jsx';

export default function AnalyzePage() {
    const [file, setFile] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    
    // Streaming states
    const [taskId, setTaskId] = useState(null);
    const [status, setStatus] = useState('idle'); // idle, working, complete, error
    const [progress, setProgress] = useState(0);
    const [traceSteps, setTraceSteps] = useState([]);
    
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleFile = (f) => {
        setFile(f);
        setResult(null);
        setError('');
        setStatus('idle');
        setTraceSteps([]);
        setProgress(0);
    };

    const resetAnalysis = () => {
        setTaskId(null);
        setFile(null);
        setResult(null);
        setError('');
        setStatus('idle');
        setTraceSteps([]);
        setProgress(0);
    };

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files[0];
        if (f) handleFile(f);
    }, []);

    const handleSubmit = async () => {
        if (!file) return;
        setStatus('working');
        setError('');
        setResult(null);
        setTraceSteps([{ step: 'initializing', status: 'pending' }]);
        setProgress(5);
        try {
            // 1. Start the task
            const data = await analysis.createTask(file);
            setTaskId(data.task_id);
            setTraceSteps(prev => [...prev, { step: 'initializing', status: 'complete' }]);
        } catch (err) {
            setError(err.message || 'Failed to start analysis task');
            setStatus('error');
        }
    };

    // 2. SSE Subscription
    useEffect(() => {
        if (!taskId || status !== 'working') return;

        const token = localStorage.getItem('token');
        const url = `/api/v1/stream/task/${taskId}?token=${token}`;
        const eventSource = new EventSource(url);

        eventSource.onopen = () => {
             console.log("SSE connected!");
             setTraceSteps(prev => [...prev, { step: 'connecting_agent', status: 'complete' }]);
        };

        eventSource.onmessage = (e) => {
            console.log("SSE raw message:", e.data);
            try {
                // Not standard JSON sometimes depending on fastAPI format, but we yield standard json in task_service
                // Oh wait, Python yields `event: {stream_event.event}\ndata: {data_str}\n\n`
                // EventSource handles the splitting. The generic event is captured here if it has no custom event name.
                // Wait, if it has a custom event name, it requires addEventListener.
            } catch (err) {
                console.error("Parse error", err);
            }
        };

        // Listen for standard updates
        const addSseListener = (name, handler) => {
            eventSource.addEventListener(name, (e) => {
                const data = JSON.parse(e.data);
                handler(data);
            });
        };

        addSseListener('running', (data) => {
            setProgress(data.progress || 10);
            if (data.trace) {
                 setTraceSteps(prev => [...prev.filter(t => t.step !== data.trace.step), data.trace]);
            }
        });

        addSseListener('agent_step', (data) => {
            if (data.progress) setProgress(data.progress);
            if (data.trace) {
                 setTraceSteps(prev => {
                     // Update existing step or append newly discovered step
                     const existing = prev.findIndex(t => t.step === data.trace.step);
                     if (existing >= 0) {
                         const next = [...prev];
                         next[existing] = data.trace;
                         return next;
                     }
                     return [...prev, data.trace];
                 });
            }
        });

        addSseListener('complete', (data) => {
            setProgress(100);
            setStatus('complete');
            
            // Format to match old sync endpoint
            setResult({
                filename: file ? file.name : "unknown",
                analysis: data.result || data
            });
            eventSource.close();
        });

        addSseListener('error', (data) => {
            setStatus('error');
            setError(data.error || 'Agent encountered a fatal error');
            eventSource.close();
        });
        
        addSseListener('done', () => {
             eventSource.close();
        });

        eventSource.onerror = (err) => {
            console.error("SSE Error:", err);
            // Don't kill it immediately on first error; it might reconnect
        };

        return () => {
            eventSource.close();
        };
    }, [taskId, status]);

    const formatSize = (bytes) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / 1048576).toFixed(1)} MB`;
    };

    const riskColor = (level) => `var(--risk-${level})`;

    return (
        <>
            <div className="page-header">
                <h2>Analyze Application</h2>
                <p>Upload a loan application to receive an AI-powered risk analysis</p>
            </div>

            {/* ── Upload Zone ── */}
            {status === 'idle' && (
                <>
                    <div
                        className={`upload-zone${dragOver ? ' drag-over' : ''}`}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                    >
                        <input
                            type="file"
                            accept=".pdf,.docx,.txt"
                            onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
                        />
                        <div className="upload-icon">📤</div>
                        <h3>Drag & drop your document here</h3>
                        <p>or <span className="highlight">click to browse</span></p>
                        <p style={{ marginTop: '8px' }}>Supports PDF, DOCX, TXT — up to 25 MB</p>
                    </div>

                    {file && (
                        <div className="file-selected">
                            <span>📎</span>
                            <span className="file-name">{file.name}</span>
                            <span className="file-size">{formatSize(file.size)}</span>
                            <button
                                className="btn btn-secondary"
                                onClick={() => setFile(null)}
                            >
                                ✕
                            </button>
                        </div>
                    )}

                    {error && <div className="alert error" style={{ marginTop: '16px' }}>⚠️ {error}</div>}

                    <div style={{ marginTop: '24px', display: 'flex', gap: '12px' }}>
                        <button
                            className="btn btn-primary btn-lg"
                            disabled={!file}
                            onClick={handleSubmit}
                        >
                            🔍 Analyze Application
                        </button>
                    </div>
                </>
            )}

            {/* ── Loading & Trace ── */}
            {status === 'working' && (
                <div className="grid-2col">
                    <div className="card">
                        <div className="loading-overlay" style={{ background: 'transparent', minHeight: '300px' }}>
                            <div className="spinner" />
                            <h3>Analyzing {file?.name}...</h3>
                            <p style={{ color: 'var(--text-muted)' }}>
                                Agentic workflow executing in the background. Streamed via SSE.
                            </p>
                            <div className="progress-bar">
                                <div className="progress-fill" style={{ width: `${progress}%` }} />
                            </div>
                        </div>
                    </div>
                    
                    <AgentTrace traceSteps={traceSteps} />
                </div>
            )}
            
            {/* ── Error ── */}
            {status === 'error' && (
                <div className="alert error" style={{ marginTop: '16px' }}>
                    ⚠️ {error}
                    <button className="btn btn-secondary" style={{ marginLeft: '16px' }} onClick={resetAnalysis}>Try Again</button>
                </div>
            )}

            {/* ── Results ── */}
            {result && (
                <>
                    <div className="alert success">✅ Analysis complete for {result.filename}</div>

                    <div className="analysis-result">
                        {/* Sidebar — Score & Recommendation */}
                        <div className="result-sidebar">
                            <div className="card">
                                <div className="risk-gauge">
                                    <div
                                        className={`risk-score-circle ${result.analysis.overall_risk_level}`}
                                        style={{ '--score': result.analysis.overall_risk_score }}
                                    >
                                        <span
                                            className="risk-score-value"
                                            style={{ color: riskColor(result.analysis.overall_risk_level) }}
                                        >
                                            {result.analysis.overall_risk_score}
                                        </span>
                                        <span className="risk-score-label">Risk Score</span>
                                    </div>
                                    <span className={`badge ${result.analysis.overall_risk_level}`}>
                                        {result.analysis.overall_risk_level.toUpperCase()} RISK
                                    </span>
                                </div>
                            </div>

                            <div className="card recommendation-card">
                                <div className="recommendation-label">Recommendation</div>
                                <div className="recommendation-value">
                                    <span
                                        className={`badge ${result.analysis.recommendation.toLowerCase().replace('_', '-')}`}
                                        style={{ fontSize: '1rem', padding: '6px 16px' }}
                                    >
                                        {result.analysis.recommendation === 'APPROVE' && '✅ '}
                                        {result.analysis.recommendation === 'DENY' && '❌ '}
                                        {result.analysis.recommendation === 'MANUAL_REVIEW' && '👁️ '}
                                        {result.analysis.recommendation.replace('_', ' ')}
                                    </span>
                                </div>
                            </div>

                            <div className="card">
                                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                                    <div style={{ marginBottom: '8px' }}>
                                        ⏱️ Processing: {result.analysis.processing_time_seconds}s
                                    </div>
                                    <div style={{ marginBottom: '8px' }}>
                                        📋 Guidelines checked: {result.analysis.guidelines_checked}
                                    </div>
                                    <div>
                                        🚩 Flags: {result.analysis.risk_flags.length}
                                    </div>
                                </div>
                            </div>

                            <button
                                className="btn btn-primary btn-block"
                                onClick={resetAnalysis}
                            >
                                🔍 Analyze Another
                            </button>
                        </div>

                        {/* Main — Details */}
                        <div className="result-main">
                            <div className="card">
                                <div className="card-header">
                                    <span className="card-title">Summary</span>
                                </div>
                                <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                                    {result.analysis.summary}
                                </p>
                            </div>

                            {result.analysis.risk_flags.length > 0 && (
                                <div className="card">
                                    <div className="card-header">
                                        <span className="card-title">
                                            Risk Flags ({result.analysis.risk_flags.length})
                                        </span>
                                    </div>
                                    <div className="risk-flags-list">
                                        {result.analysis.risk_flags.map((flag, i) => (
                                            <div key={i} className={`risk-flag-item ${flag.severity}`}>
                                                <div className="risk-flag-header">
                                                    <span className="risk-flag-category">{flag.category}</span>
                                                    <span className={`badge ${flag.severity}`}>{flag.severity}</span>
                                                </div>
                                                <div className="risk-flag-description">{flag.description}</div>
                                                <div className="risk-flag-guideline">
                                                    📌 {flag.guideline_reference}
                                                </div>
                                                <div className="confidence-bar">
                                                    <div
                                                        className="confidence-fill"
                                                        style={{ width: `${(flag.confidence * 100).toFixed(0)}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div className="card">
                                <div className="card-header">
                                    <span className="card-title">Detailed Analysis</span>
                                </div>
                                <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                                    {result.analysis.detailed_analysis}
                                </p>
                            </div>
                        </div>
                    </div>
                </>
            )}
        </>
    );
}
