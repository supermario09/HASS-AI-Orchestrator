/**
 * EntityWatchlist — lets users pin specific HA entities and watch their live state.
 *
 * - Persists selections to localStorage (no backend needed)
 * - Auto-refreshes entity states every 15s
 * - Groups entities by domain with colour coding
 * - Clicking an entity chip shows its attributes in an expandable panel
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Search, Plus, X, RefreshCw, Loader, ChevronDown, ChevronUp } from 'lucide-react';

const LS_KEY = 'entity_watchlist_v1';

// Domain colour palette — same as AgentDetails
const DOMAIN_COLOURS = {
    light:               'bg-yellow-500/20 border-yellow-500/40 text-yellow-300',
    switch:              'bg-blue-500/20 border-blue-500/40 text-blue-300',
    climate:             'bg-orange-500/20 border-orange-500/40 text-orange-300',
    sensor:              'bg-purple-500/20 border-purple-500/40 text-purple-300',
    binary_sensor:       'bg-cyan-500/20 border-cyan-500/40 text-cyan-300',
    media_player:        'bg-pink-500/20 border-pink-500/40 text-pink-300',
    camera:              'bg-green-500/20 border-green-500/40 text-green-300',
    cover:               'bg-amber-500/20 border-amber-500/40 text-amber-300',
    lock:                'bg-red-500/20 border-red-500/40 text-red-300',
    fan:                 'bg-sky-500/20 border-sky-500/40 text-sky-300',
    input_boolean:       'bg-indigo-500/20 border-indigo-500/40 text-indigo-300',
    alarm_control_panel: 'bg-rose-500/20 border-rose-500/40 text-rose-300',
    vacuum:              'bg-teal-500/20 border-teal-500/40 text-teal-300',
    number:              'bg-lime-500/20 border-lime-500/40 text-lime-300',
    input_number:        'bg-lime-500/20 border-lime-500/40 text-lime-300',
    weather:             'bg-sky-500/20 border-sky-500/40 text-sky-300',
    person:              'bg-fuchsia-500/20 border-fuchsia-500/40 text-fuchsia-300',
};

const domainColour = (eid) =>
    DOMAIN_COLOURS[eid?.split('.')[0]] ?? 'bg-slate-700/50 border-slate-600 text-slate-300';

const stateColour = (state) => {
    if (!state) return 'text-slate-400';
    const s = state.toLowerCase();
    if (s === 'on' || s === 'home' || s === 'open' || s === 'unlocked') return 'text-green-400';
    if (s === 'off' || s === 'away' || s === 'closed' || s === 'locked') return 'text-slate-400';
    if (s === 'unavailable' || s === 'unknown') return 'text-red-400/70';
    return 'text-blue-300';
};

export function EntityWatchlist() {
    const [pinned, setPinned]           = useState(() => {
        try { return JSON.parse(localStorage.getItem(LS_KEY) || '[]'); }
        catch { return []; }
    });
    const [allEntities, setAllEntities] = useState([]);
    const [liveStates, setLiveStates]   = useState({});   // entity_id → {state, attributes, friendly_name}
    const [search, setSearch]           = useState('');
    const [showDrop, setShowDrop]       = useState(false);
    const [loadingAll, setLoadingAll]   = useState(false);
    const [refreshing, setRefreshing]   = useState(false);
    const [expanded, setExpanded]       = useState({});   // entity_id → bool
    const searchRef = useRef(null);
    const dropRef   = useRef(null);

    // Persist pinned list to localStorage
    useEffect(() => {
        localStorage.setItem(LS_KEY, JSON.stringify(pinned));
    }, [pinned]);

    // Load full entity catalog once
    useEffect(() => {
        setLoadingAll(true);
        fetch('api/entities')
            .then(r => r.json())
            .then(d => setAllEntities(Array.isArray(d) ? d : []))
            .catch(() => setAllEntities([]))
            .finally(() => setLoadingAll(false));
    }, []);

    // Refresh live states — called on mount and every 15s
    const refreshStates = useCallback(async (silent = false) => {
        if (!silent) setRefreshing(true);
        try {
            const data = await fetch('api/entities').then(r => r.json());
            if (!Array.isArray(data)) return;
            const map = {};
            data.forEach(e => {
                map[e.entity_id] = {
                    state: e.state,
                    friendly_name: e.friendly_name,
                    attributes: e.attributes || {},
                };
            });
            setLiveStates(map);
            // Also update allEntities in case new ones appeared
            setAllEntities(data);
        } catch (_) {}
        finally { if (!silent) setRefreshing(false); }
    }, []);

    useEffect(() => {
        refreshStates(true);
        const id = setInterval(() => refreshStates(true), 15000);
        return () => clearInterval(id);
    }, [refreshStates]);

    // Close dropdown on outside click
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

    const pinnedSet = new Set(pinned);

    const filtered = search.trim().length >= 1
        ? allEntities
              .filter(e =>
                  !pinnedSet.has(e.entity_id) &&
                  (e.entity_id.toLowerCase().includes(search.toLowerCase()) ||
                   (e.friendly_name || '').toLowerCase().includes(search.toLowerCase()))
              )
              .slice(0, 40)
        : [];

    const addEntity = (eid) => {
        if (!pinnedSet.has(eid)) setPinned(prev => [...prev, eid]);
        setSearch('');
        setShowDrop(false);
    };

    const removeEntity = (eid) => setPinned(prev => prev.filter(e => e !== eid));

    const toggleExpand = (eid) => setExpanded(prev => ({ ...prev, [eid]: !prev[eid] }));

    // Group pinned by domain
    const grouped = pinned.reduce((acc, eid) => {
        const domain = eid.split('.')[0];
        if (!acc[domain]) acc[domain] = [];
        acc[domain].push(eid);
        return acc;
    }, {});

    return (
        <div className="space-y-6">
            {/* ── Header ── */}
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-xl font-semibold text-white">Entity Watchlist</h2>
                    <p className="text-slate-500 text-sm mt-0.5">
                        Pin any Home Assistant entity to monitor its live state.
                        Selections are saved in your browser.
                    </p>
                </div>
                <button
                    onClick={() => refreshStates(false)}
                    disabled={refreshing}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-400 hover:text-white text-sm transition-colors"
                    title="Refresh states"
                >
                    <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
                    Refresh
                </button>
            </div>

            {/* ── Search / Add ── */}
            <div className="relative">
                <div className="flex items-center gap-2 bg-slate-800 border border-slate-600 rounded-xl px-3 py-2.5 focus-within:border-blue-500 transition-colors">
                    <Search size={15} className="text-slate-400 shrink-0" />
                    <input
                        ref={searchRef}
                        type="text"
                        placeholder={loadingAll ? 'Loading entities…' : 'Search by entity ID or friendly name…'}
                        value={search}
                        disabled={loadingAll}
                        onChange={e => { setSearch(e.target.value); setShowDrop(true); }}
                        onFocus={() => search.trim() && setShowDrop(true)}
                        className="flex-1 bg-transparent text-slate-200 placeholder-slate-500 text-sm focus:outline-none"
                    />
                    {loadingAll && <Loader size={14} className="animate-spin text-slate-500 shrink-0" />}
                </div>

                {showDrop && filtered.length > 0 && (
                    <div ref={dropRef}
                        className="absolute z-20 w-full mt-1 bg-slate-800 border border-slate-600 rounded-xl shadow-2xl max-h-72 overflow-y-auto">
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
                                <span className={`text-xs shrink-0 ${stateColour(e.state)}`}>{e.state}</span>
                                <Plus size={14} className="text-slate-600 group-hover:text-blue-400 shrink-0 transition-colors" />
                            </button>
                        ))}
                    </div>
                )}

                {showDrop && search.trim().length >= 1 && filtered.length === 0 && !loadingAll && (
                    <div className="absolute z-20 w-full mt-1 bg-slate-800 border border-slate-600 rounded-xl px-4 py-3 text-slate-500 text-sm">
                        No matching entities found.
                    </div>
                )}
            </div>

            {/* ── Pinned entity cards ── */}
            {pinned.length === 0 ? (
                <div className="border-2 border-dashed border-slate-700 rounded-xl p-12 text-center">
                    <Search size={28} className="text-slate-600 mx-auto mb-3" />
                    <p className="text-slate-400 font-medium">No entities pinned yet</p>
                    <p className="text-slate-600 text-sm mt-1">
                        Search above and click any entity to add it to your watchlist.
                    </p>
                </div>
            ) : (
                <div className="space-y-6">
                    {Object.entries(grouped).map(([domain, eids]) => (
                        <div key={domain}>
                            {/* Domain group header */}
                            <div className="flex items-center gap-2 mb-3">
                                <span className={`text-[10px] px-2 py-0.5 rounded border font-mono font-bold uppercase ${domainColour(domain + '.x')}`}>
                                    {domain}
                                </span>
                                <span className="text-xs text-slate-600">{eids.length} entities</span>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                {eids.map(eid => {
                                    const live = liveStates[eid];
                                    const name = live?.friendly_name || eid;
                                    const state = live?.state ?? '…';
                                    const attrs = live?.attributes || {};
                                    const isExpanded = expanded[eid];
                                    const attrEntries = Object.entries(attrs)
                                        .filter(([k]) => !['entity_picture', 'icon', 'supported_features', 'assumed_state'].includes(k))
                                        .slice(0, 20);

                                    return (
                                        <div key={eid} className={`rounded-xl border p-4 ${domainColour(eid)} group`}>
                                            {/* Card header */}
                                            <div className="flex justify-between items-start gap-2">
                                                <div className="min-w-0 flex-1">
                                                    <div className="text-sm font-medium truncate" title={eid}>
                                                        {name !== eid ? name : eid.split('.')[1].replace(/_/g, ' ')}
                                                    </div>
                                                    <div className="text-[10px] opacity-50 font-mono truncate">{eid}</div>
                                                </div>
                                                <button
                                                    onClick={() => removeEntity(eid)}
                                                    className="opacity-0 group-hover:opacity-60 hover:!opacity-100 transition-opacity p-0.5 rounded hover:bg-red-500/20"
                                                    title="Remove from watchlist"
                                                >
                                                    <X size={13} className="text-red-400" />
                                                </button>
                                            </div>

                                            {/* State value */}
                                            <div className={`text-2xl font-bold mt-2 mb-1 ${stateColour(state)}`}>
                                                {state}
                                                {attrs.unit_of_measurement && (
                                                    <span className="text-sm font-normal opacity-60 ml-1">{attrs.unit_of_measurement}</span>
                                                )}
                                            </div>

                                            {/* Expand / collapse attributes */}
                                            {attrEntries.length > 0 && (
                                                <>
                                                    <button
                                                        onClick={() => toggleExpand(eid)}
                                                        className="flex items-center gap-1 text-[10px] opacity-50 hover:opacity-80 transition-opacity mt-1"
                                                    >
                                                        {isExpanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
                                                        {isExpanded ? 'Hide' : 'Show'} attributes
                                                    </button>

                                                    {isExpanded && (
                                                        <div className="mt-2 space-y-0.5 border-t border-current/10 pt-2">
                                                            {attrEntries.map(([k, v]) => (
                                                                <div key={k} className="flex justify-between text-[10px] gap-2">
                                                                    <span className="opacity-50 truncate">{k}</span>
                                                                    <span className="opacity-80 truncate max-w-[50%] text-right font-mono">
                                                                        {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                                                                    </span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    )}
                                                </>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ))}

                    {/* Clear all */}
                    <div className="text-center pt-2">
                        <button
                            onClick={() => { setPinned([]); setExpanded({}); }}
                            className="text-xs text-red-400/60 hover:text-red-400 transition-colors"
                        >
                            Clear all pinned entities
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
