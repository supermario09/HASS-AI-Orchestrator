import React, { useState, useEffect, useRef } from 'react';
import { X, Trash2, Edit2, Check, ExternalLink, Activity, Save, Plus, Search, Loader, ThumbsUp, ThumbsDown } from 'lucide-react';

// Domain colour map for entity chips
const DOMAIN_COLOURS = {
    light:              'bg-yellow-500/20 border-yellow-500/40 text-yellow-300',
    switch:             'bg-blue-500/20 border-blue-500/40 text-blue-300',
    climate:            'bg-orange-500/20 border-orange-500/40 text-orange-300',
    sensor:             'bg-purple-500/20 border-purple-500/40 text-purple-300',
    binary_sensor:      'bg-cyan-500/20 border-cyan-500/40 text-cyan-300',
    media_player:       'bg-pink-500/20 border-pink-500/40 text-pink-300',
    camera:             'bg-green-500/20 border-green-500/40 text-green-300',
    cover:              'bg-amber-500/20 border-amber-500/40 text-amber-300',
    lock:               'bg-red-500/20 border-red-500/40 text-red-300',
    fan:                'bg-sky-500/20 border-sky-500/40 text-sky-300',
    input_boolean:      'bg-indigo-500/20 border-indigo-500/40 text-indigo-300',
    alarm_control_panel:'bg-rose-500/20 border-rose-500/40 text-rose-300',
    vacuum:             'bg-teal-500/20 border-teal-500/40 text-teal-300',
};
const domainColour = (eid) =>
    DOMAIN_COLOURS[eid?.split('.')[0]] ?? 'bg-slate-700/50 border-slate-600 text-slate-300';

// ── Entity picker component ────────────────────────────────────────────────────
function EntityPicker({ entities, onChange }) {
    const [search, setSearch]           = useState('');
    const [allEntities, setAllEntities] = useState([]);
    const [loadingAll, setLoadingAll]   = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const [savingEntities, setSavingEntities] = useState(false);
    const [savedOk, setSavedOk]         = useState(false);
    const [saveError, setSaveError]     = useState(null);
    const inputRef  = useRef(null);
    const dropRef   = useRef(null);

    // Load full entity list from HA once
    useEffect(() => {
        setLoadingAll(true);
        fetch('api/entities')
            .then(r => r.json())
            .then(data => { setAllEntities(Array.isArray(data) ? data : []); })
            .catch(() => setAllEntities([]))
            .finally(() => setLoadingAll(false));
    }, []);

    // Close dropdown on outside click
    useEffect(() => {
        const handler = (e) => {
            if (dropRef.current && !dropRef.current.contains(e.target) &&
                inputRef.current  && !inputRef.current.contains(e.target)) {
                setShowDropdown(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    const entitySet = new Set(entities);

    const filtered = search.trim().length >= 1
        ? allEntities
              .filter(e =>
                  !entitySet.has(e.entity_id) &&
                  (e.entity_id.toLowerCase().includes(search.toLowerCase()) ||
                   (e.friendly_name || '').toLowerCase().includes(search.toLowerCase()))
              )
              .slice(0, 30)
        : [];

    const addEntity = (eid) => {
        onChange([...entities, eid]);
        setSearch('');
        setShowDropdown(false);
        setSavedOk(false);
        setSaveError(null);
    };

    const removeEntity = (eid) => {
        onChange(entities.filter(e => e !== eid));
        setSavedOk(false);
        setSaveError(null);
    };

    // Expose a save method via ref (called by parent's save button)
    // We'll pass onSave as a prop instead
    return { addEntity, removeEntity, filtered, search, setSearch, showDropdown, setShowDropdown,
             inputRef, dropRef, loadingAll, savingEntities, savedOk, saveError, entitySet };
}

// ── Main component ─────────────────────────────────────────────────────────────
const AgentDetails = ({ agent, onClose, onDelete }) => {
    const [activeTab, setActiveTab]       = useState('overview');
    const [decisions, setDecisions]       = useState([]);
    const [isEditing, setIsEditing]       = useState(false);
    const [instruction, setInstruction]   = useState(agent?.instruction || '');
    const [name, setName]                 = useState(agent?.name || '');
    const [interval, setIntervalValue]    = useState(agent?.decision_interval || 120);
    const [loading, setLoading]           = useState(false);

    // Entity editor state
    const [entities, setEntities]         = useState(agent?.entities ?? []);
    const [allEntities, setAllEntities]   = useState([]);
    const [loadingAll, setLoadingAll]     = useState(false);
    const [entitySearch, setEntitySearch] = useState('');
    const [showDrop, setShowDrop]         = useState(false);
    const [entitySaving, setEntitySaving] = useState(false);
    const [entitySaved, setEntitySaved]   = useState(false);
    const [entityError, setEntityError]   = useState(null);
    const searchRef = useRef(null);
    const dropRef   = useRef(null);

    // Reset on agent change
    useEffect(() => {
        if (!agent?.agent_id) return;
        fetchDecisions();
        setInstruction(agent.instruction || '');
        setName(agent.name || '');
        setIntervalValue(agent.decision_interval ? parseInt(agent.decision_interval) : 120);
        setEntities(agent.entities ?? []);
        setEntitySaved(false);
        setEntityError(null);
    }, [agent]);

    // Load all HA entities once
    useEffect(() => {
        setLoadingAll(true);
        fetch('api/entities')
            .then(r => r.json())
            .then(d => setAllEntities(Array.isArray(d) ? d : []))
            .catch(() => setAllEntities([]))
            .finally(() => setLoadingAll(false));
    }, []);

    // Close autocomplete on outside click
    useEffect(() => {
        const handler = (e) => {
            if (dropRef.current  && !dropRef.current.contains(e.target) &&
                searchRef.current && !searchRef.current.contains(e.target)) {
                setShowDrop(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    const fetchDecisions = async () => {
        try {
            const res  = await fetch(`api/decisions?agent_id=${agent.agent_id}&limit=20`);
            const data = await res.json();
            setDecisions(data);
        } catch (e) { console.error('Failed to fetch decisions', e); }
    };

    // ── instruction / name / interval save ──────────────────────────────────
    const handleUpdate = async () => {
        setLoading(true);
        try {
            const res = await fetch(`api/factory/agents/${agent.agent_id}`, {
                method:  'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    instruction:       instruction,
                    name:              name,
                    decision_interval: parseInt(interval),
                }),
            });
            if (res.ok) {
                setIsEditing(false);
                if (name !== agent.name) window.location.reload();
            } else {
                alert('Failed to update agent');
            }
        } catch (e) {
            console.error(e);
            alert('Error updating agent');
        } finally {
            setLoading(false);
        }
    };

    // ── entity save ──────────────────────────────────────────────────────────
    const handleSaveEntities = async () => {
        setEntitySaving(true);
        setEntitySaved(false);
        setEntityError(null);
        try {
            const res = await fetch(`api/factory/agents/${agent.agent_id}`, {
                method:  'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ entities }),
            });
            if (res.ok) {
                setEntitySaved(true);
                setTimeout(() => setEntitySaved(false), 3000);
            } else {
                const err = await res.text();
                setEntityError(`Save failed: ${err}`);
            }
        } catch (e) {
            setEntityError(`Save failed: ${e.message}`);
        } finally {
            setEntitySaving(false);
        }
    };

    // ── entity add / remove ──────────────────────────────────────────────────
    const entitySet = new Set(entities);

    const addEntity = (eid) => {
        if (!entitySet.has(eid)) {
            setEntities(prev => [...prev, eid]);
        }
        setEntitySearch('');
        setShowDrop(false);
        setEntitySaved(false);
        setEntityError(null);
    };

    const removeEntity = (eid) => {
        setEntities(prev => prev.filter(e => e !== eid));
        setEntitySaved(false);
        setEntityError(null);
    };

    const filtered = entitySearch.trim().length >= 1
        ? allEntities
              .filter(e =>
                  !entitySet.has(e.entity_id) &&
                  (e.entity_id.toLowerCase().includes(entitySearch.toLowerCase()) ||
                   (e.friendly_name || '').toLowerCase().includes(entitySearch.toLowerCase()))
              )
              .slice(0, 30)
        : [];

    // ── decision feedback ────────────────────────────────────────────────────
    // Map of timestamp → "up" | "down" | null
    const [feedbackMap, setFeedbackMap] = useState({});

    // Pre-populate feedback from loaded decisions (if already rated)
    useEffect(() => {
        const initial = {};
        decisions.forEach(d => {
            if (d.feedback) initial[d.timestamp] = d.feedback;
        });
        setFeedbackMap(initial);
    }, [decisions]);

    const submitFeedback = async (timestamp, feedback) => {
        // Toggle off if already selected
        const current = feedbackMap[timestamp];
        const next = current === feedback ? null : feedback;
        setFeedbackMap(prev => ({ ...prev, [timestamp]: next }));

        if (next) {
            try {
                await fetch('api/decisions/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ agent_id: agent.agent_id, timestamp, feedback: next }),
                });
            } catch (e) {
                console.error('Failed to submit feedback', e);
            }
        }
    };

    // ── delete ───────────────────────────────────────────────────────────────
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

    const confirmDelete = async () => {
        setLoading(true);
        try {
            const res = await fetch(`api/factory/agents/${agent.agent_id}`, { method: 'DELETE' });
            if (res.ok) { onDelete(agent.agent_id); onClose(); window.location.reload(); }
            else alert('Failed to delete agent.');
        } catch (e) { alert('Failed to delete agent.'); }
        finally { setLoading(false); setShowDeleteConfirm(false); }
    };

    if (!agent) return null;

    return (
        <div className="fixed inset-0 z-50 overflow-hidden">
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

            <div className="absolute inset-y-0 right-0 w-full max-w-2xl bg-slate-900 border-l border-slate-700 shadow-2xl flex flex-col">

                {/* ── Header ── */}
                <div className="p-6 border-b border-slate-700 flex justify-between items-start bg-slate-800/50">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <div className={`w-3 h-3 rounded-full ${agent.status === 'deciding' ? 'bg-green-500 animate-pulse' : 'bg-slate-500'}`} />
                            {isEditing ? (
                                <input
                                    type="text" value={name}
                                    onChange={e => setName(e.target.value)}
                                    className="bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xl font-bold text-white focus:outline-none focus:border-blue-500 w-full"
                                />
                            ) : (
                                <h2 className="text-2xl font-bold text-white">{agent.name}</h2>
                            )}
                            <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-300 font-mono">{agent.model}</span>
                        </div>
                        <p className="text-slate-400 text-sm">ID: {agent.agent_id}</p>
                    </div>
                    <div className="flex gap-2">
                        <button onClick={() => setShowDeleteConfirm(true)}
                            className="p-2 text-red-400 hover:bg-red-400/10 rounded-lg transition-colors" title="Delete Agent">
                            <Trash2 size={20} />
                        </button>
                        <button onClick={onClose}
                            className="p-2 text-slate-400 hover:bg-slate-700/50 rounded-lg transition-colors">
                            <X size={24} />
                        </button>
                    </div>
                </div>

                {/* ── Delete confirm overlay ── */}
                {showDeleteConfirm && (
                    <div className="absolute inset-0 z-10 bg-slate-900/90 flex items-center justify-center p-6 backdrop-blur-sm">
                        <div className="bg-slate-800 border border-red-500/30 rounded-xl p-6 max-w-sm w-full shadow-2xl">
                            <div className="flex flex-col items-center text-center">
                                <div className="w-12 h-12 rounded-full bg-red-500/20 text-red-500 flex items-center justify-center mb-4">
                                    <Trash2 size={24} />
                                </div>
                                <h3 className="text-xl font-bold text-white mb-2">Delete Agent?</h3>
                                <p className="text-slate-400 mb-6">
                                    Are you sure you want to permanently delete <strong>{agent.name}</strong>? This cannot be undone.
                                </p>
                                <div className="flex gap-3 w-full">
                                    <button onClick={() => setShowDeleteConfirm(false)}
                                        className="flex-1 py-2 px-4 rounded-lg bg-slate-700 text-slate-200 hover:bg-slate-600 border border-slate-600 transition-colors font-medium">
                                        Cancel
                                    </button>
                                    <button onClick={confirmDelete} disabled={loading}
                                        className="flex-1 py-2 px-4 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-colors font-medium flex items-center justify-center gap-2">
                                        {loading ? 'Deleting…' : 'Delete Forever'}
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* ── Tabs ── */}
                <div className="flex border-b border-slate-700 px-6">
                    {[['overview','Overview'],['entities','Entities'],['history','History'],['json','Config']].map(([id,label]) => (
                        <button key={id}
                            className={`py-4 px-4 text-sm font-medium border-b-2 transition-colors ${activeTab === id ? 'border-blue-500 text-blue-400' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
                            onClick={() => setActiveTab(id)}>
                            {label}
                            {id === 'entities' && (
                                <span className="ml-1.5 text-xs px-1.5 py-0.5 rounded-full bg-slate-700 text-slate-400">
                                    {entities.length}
                                </span>
                            )}
                        </button>
                    ))}
                </div>

                {/* ── Content ── */}
                <div className="flex-1 overflow-y-auto p-6">

                    {/* ── OVERVIEW tab ── */}
                    {activeTab === 'overview' && (
                        <div className="space-y-6">
                            <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                        <Activity size={18} className="text-blue-400" /> Primary Instruction
                                    </h3>
                                    {!isEditing ? (
                                        <button onClick={() => setIsEditing(true)}
                                            className="text-xs flex items-center gap-1 text-slate-400 hover:text-white bg-slate-700/50 px-2 py-1 rounded transition-colors">
                                            <Edit2 size={12} /> Edit
                                        </button>
                                    ) : (
                                        <div className="flex gap-2">
                                            <button onClick={() => setIsEditing(false)} className="text-xs text-slate-400 hover:text-white">Cancel</button>
                                            <button onClick={handleUpdate} disabled={loading}
                                                className="text-xs flex items-center gap-1 text-green-400 hover:text-green-300 bg-green-400/10 px-2 py-1 rounded transition-colors">
                                                <Save size={12} /> {loading ? 'Saving…' : 'Save'}
                                            </button>
                                        </div>
                                    )}
                                </div>

                                {isEditing ? (
                                    <div className="space-y-4">
                                        <textarea value={instruction} onChange={e => setInstruction(e.target.value)}
                                            className="w-full h-48 bg-slate-900/50 border border-slate-600 rounded-lg p-3 text-slate-200 focus:outline-none focus:border-blue-500 font-mono text-sm leading-relaxed" />
                                        <div className="flex items-center gap-4 bg-slate-950 p-4 rounded-lg border border-slate-700">
                                            <div className="text-slate-200 text-sm font-medium whitespace-nowrap">Decision Interval:</div>
                                            <div className="text-blue-400 font-mono w-12 text-right">{interval}s</div>
                                            <input type="range" min="5" max="600" step="5" value={interval}
                                                onChange={e => setIntervalValue(parseInt(e.target.value))}
                                                className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
                                            <div className="text-[10px] text-slate-500 w-8 text-right">600s</div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="space-y-4">
                                        <div className="prose prose-invert max-w-none text-slate-300 text-sm whitespace-pre-wrap font-mono bg-slate-900/30 p-4 rounded-lg border border-slate-700/30">
                                            {agent.instruction || instruction}
                                        </div>
                                        <div className="flex items-center gap-2 text-xs text-slate-400 px-1">
                                            <Check size={12} className="text-blue-500" />
                                            Updating every <span className="text-slate-200 font-bold">{agent.decision_interval || interval} seconds</span>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Quick entity summary on overview */}
                            <div className="bg-slate-800/30 rounded-xl p-4 border border-slate-700/50">
                                <div className="flex justify-between items-center mb-3">
                                    <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                                        <ExternalLink size={14} className="text-purple-400" />
                                        Assigned Entities <span className="text-slate-500">({entities.length})</span>
                                    </h3>
                                    <button onClick={() => setActiveTab('entities')}
                                        className="text-xs text-purple-400 hover:text-purple-300 transition-colors">
                                        Manage →
                                    </button>
                                </div>
                                {entities.length > 0 ? (
                                    <div className="flex flex-wrap gap-1.5">
                                        {entities.slice(0, 8).map(eid => (
                                            <span key={eid} className={`text-xs px-2 py-0.5 rounded-full border font-mono ${domainColour(eid)}`}>
                                                {eid}
                                            </span>
                                        ))}
                                        {entities.length > 8 && (
                                            <span className="text-xs px-2 py-0.5 rounded-full border border-slate-600 text-slate-400">
                                                +{entities.length - 8} more
                                            </span>
                                        )}
                                    </div>
                                ) : (
                                    <p className="text-slate-500 text-xs italic">No entities assigned — agent uses auto-discovery. Click Manage to add specific entities.</p>
                                )}
                            </div>
                        </div>
                    )}

                    {/* ── ENTITIES tab ── */}
                    {activeTab === 'entities' && (
                        <div className="space-y-5">
                            {/* Header + save */}
                            <div className="flex justify-between items-center">
                                <div>
                                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                        <ExternalLink size={18} className="text-purple-400" /> Entity Manager
                                    </h3>
                                    <p className="text-slate-500 text-xs mt-0.5">
                                        Entities listed here are fetched on every decision cycle and sent to the LLM as context.
                                        Empty = auto-discovery.
                                    </p>
                                </div>
                                <button
                                    onClick={handleSaveEntities}
                                    disabled={entitySaving}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all
                                        ${entitySaved
                                            ? 'bg-green-500/20 border border-green-500/40 text-green-400'
                                            : 'bg-purple-600 hover:bg-purple-500 text-white'
                                        }`}
                                >
                                    {entitySaving ? <Loader size={14} className="animate-spin" /> : <Save size={14} />}
                                    {entitySaving ? 'Saving…' : entitySaved ? '✓ Saved' : 'Save Entities'}
                                </button>
                            </div>

                            {entityError && (
                                <div className="text-red-400 text-xs bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2">
                                    {entityError}
                                </div>
                            )}

                            {/* Search / add */}
                            <div className="relative">
                                <div className="flex items-center gap-2 bg-slate-800 border border-slate-600 rounded-xl px-3 py-2.5 focus-within:border-purple-500 transition-colors">
                                    <Search size={15} className="text-slate-400 shrink-0" />
                                    <input
                                        ref={searchRef}
                                        type="text"
                                        placeholder={loadingAll ? 'Loading entities from HA…' : 'Search entity ID or friendly name…'}
                                        value={entitySearch}
                                        disabled={loadingAll}
                                        onChange={e => { setEntitySearch(e.target.value); setShowDrop(true); }}
                                        onFocus={() => entitySearch.trim() && setShowDrop(true)}
                                        className="flex-1 bg-transparent text-slate-200 placeholder-slate-500 text-sm focus:outline-none"
                                    />
                                    {loadingAll && <Loader size={14} className="animate-spin text-slate-500 shrink-0" />}
                                </div>

                                {/* Dropdown */}
                                {showDrop && filtered.length > 0 && (
                                    <div ref={dropRef}
                                        className="absolute z-20 w-full mt-1 bg-slate-800 border border-slate-600 rounded-xl shadow-2xl max-h-64 overflow-y-auto">
                                        {filtered.map(e => (
                                            <button
                                                key={e.entity_id}
                                                onMouseDown={() => addEntity(e.entity_id)}
                                                className="w-full flex items-center gap-3 px-3 py-2.5 hover:bg-slate-700/50 transition-colors text-left group"
                                            >
                                                <span className={`shrink-0 text-[10px] px-1.5 py-0.5 rounded border font-mono ${domainColour(e.entity_id)}`}>
                                                    {e.entity_id.split('.')[0]}
                                                </span>
                                                <div className="flex-1 min-w-0">
                                                    <div className="text-slate-200 text-sm font-mono truncate">{e.entity_id}</div>
                                                    {e.friendly_name && e.friendly_name !== e.entity_id && (
                                                        <div className="text-slate-500 text-xs truncate">{e.friendly_name}</div>
                                                    )}
                                                </div>
                                                <span className="text-slate-600 text-xs shrink-0">{e.state}</span>
                                                <Plus size={14} className="text-slate-600 group-hover:text-purple-400 shrink-0 transition-colors" />
                                            </button>
                                        ))}
                                    </div>
                                )}

                                {showDrop && entitySearch.trim().length >= 1 && filtered.length === 0 && !loadingAll && (
                                    <div className="absolute z-20 w-full mt-1 bg-slate-800 border border-slate-600 rounded-xl px-4 py-3 text-slate-500 text-sm">
                                        No matching entities found.
                                    </div>
                                )}
                            </div>

                            {/* Current entity chips */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">
                                        Assigned ({entities.length})
                                    </p>
                                    {entities.length > 0 && (
                                        <button
                                            onClick={() => { setEntities([]); setEntitySaved(false); }}
                                            className="text-xs text-red-400/70 hover:text-red-400 transition-colors"
                                        >
                                            Clear all
                                        </button>
                                    )}
                                </div>

                                {entities.length === 0 ? (
                                    <div className="border-2 border-dashed border-slate-700 rounded-xl p-8 text-center">
                                        <ExternalLink size={24} className="text-slate-600 mx-auto mb-2" />
                                        <p className="text-slate-500 text-sm">No entities assigned</p>
                                        <p className="text-slate-600 text-xs mt-1">
                                            Search above to add entities, or leave empty for auto-discovery.
                                        </p>
                                    </div>
                                ) : (
                                    <div className="space-y-1.5 max-h-80 overflow-y-auto pr-1">
                                        {entities.map(eid => {
                                            const meta = allEntities.find(e => e.entity_id === eid);
                                            return (
                                                <div key={eid}
                                                    className={`flex items-center justify-between gap-2 px-3 py-2 rounded-lg border ${domainColour(eid)} group`}>
                                                    <div className="flex items-center gap-2 min-w-0">
                                                        <span className="text-[10px] uppercase font-bold opacity-70 shrink-0">
                                                            {eid.split('.')[0]}
                                                        </span>
                                                        <div className="min-w-0">
                                                            <div className="text-sm font-mono truncate">{eid}</div>
                                                            {meta?.friendly_name && meta.friendly_name !== eid && (
                                                                <div className="text-xs opacity-60 truncate">{meta.friendly_name}</div>
                                                            )}
                                                        </div>
                                                    </div>
                                                    <div className="flex items-center gap-2 shrink-0">
                                                        {meta && (
                                                            <span className="text-xs opacity-50">{meta.state}</span>
                                                        )}
                                                        <button
                                                            onClick={() => removeEntity(eid)}
                                                            className="opacity-40 hover:opacity-100 transition-opacity p-0.5 rounded hover:bg-red-500/20"
                                                            title="Remove entity"
                                                        >
                                                            <X size={14} className="text-red-400" />
                                                        </button>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>

                            {/* Save footer reminder */}
                            {entities.length > 0 && !entitySaved && (
                                <p className="text-xs text-slate-500 text-center">
                                    Changes are not saved until you click <strong className="text-slate-300">Save Entities</strong>.
                                    Saved changes are written to <code className="text-purple-400">agents.yaml</code> and take effect immediately.
                                </p>
                            )}
                        </div>
                    )}

                    {/* ── HISTORY tab ── */}
                    {activeTab === 'history' && (
                        <div className="space-y-4">
                            {/* Legend */}
                            <div className="flex items-center gap-3 text-xs text-slate-500 px-1">
                                <ThumbsUp size={12} className="text-green-500" />
                                <span>Good decision</span>
                                <ThumbsDown size={12} className="text-red-400 ml-3" />
                                <span>Bad decision — your ratings help fine-tune the AI</span>
                            </div>

                            {decisions.length === 0 ? (
                                <div className="text-center text-slate-500 py-10">No recent history found.</div>
                            ) : decisions.map((decision, idx) => {
                                const ts = decision.timestamp;
                                const fb = feedbackMap[ts];
                                return (
                                    <div key={idx} className={`bg-slate-800/40 p-4 rounded-lg border text-sm transition-colors
                                        ${fb === 'up' ? 'border-green-500/40 bg-green-500/5' : fb === 'down' ? 'border-red-500/40 bg-red-500/5' : 'border-slate-700/50'}`}>
                                        <div className="flex justify-between items-start mb-2">
                                            <div className="text-xs text-slate-500">
                                                {new Date(ts).toLocaleString()}
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span className={`text-xs ${decision.dry_run ? 'text-amber-500' : 'text-slate-500'}`}>
                                                    {decision.dry_run ? 'DRY RUN' : 'LIVE'}
                                                </span>
                                                {/* Feedback buttons */}
                                                <button
                                                    onClick={() => submitFeedback(ts, 'up')}
                                                    title="Good decision"
                                                    className={`p-1 rounded transition-colors ${fb === 'up' ? 'text-green-400 bg-green-500/20' : 'text-slate-600 hover:text-green-400 hover:bg-green-500/10'}`}>
                                                    <ThumbsUp size={14} />
                                                </button>
                                                <button
                                                    onClick={() => submitFeedback(ts, 'down')}
                                                    title="Bad decision"
                                                    className={`p-1 rounded transition-colors ${fb === 'down' ? 'text-red-400 bg-red-500/20' : 'text-slate-600 hover:text-red-400 hover:bg-red-500/10'}`}>
                                                    <ThumbsDown size={14} />
                                                </button>
                                            </div>
                                        </div>
                                        <div className="text-slate-300 mb-2 font-medium">
                                            {decision.decision?.reasoning || decision.reasoning || 'No reasoning provided.'}
                                        </div>
                                        {(decision.decision?.actions || decision.action) && (
                                            <div className="bg-black/20 p-2 rounded text-xs font-mono text-blue-300 overflow-x-auto">
                                                {JSON.stringify(decision.decision?.actions || decision.action, null, 2)}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}

                    {/* ── JSON tab ── */}
                    {activeTab === 'json' && (
                        <pre className="text-xs font-mono text-slate-400 bg-black/30 p-4 rounded-lg overflow-x-auto">
                            {JSON.stringify({ ...agent, entities }, null, 2)}
                        </pre>
                    )}
                </div>
            </div>
        </div>
    );
};

export default AgentDetails;
